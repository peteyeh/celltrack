import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six
import yaml

from collections import OrderedDict
from radiomics import featureextractor
from scipy import stats
from SimpleITK import GetImageFromArray

from displaytools import *
from improcessing import *

def extract_features(image, mask_image):
    num_components, mask_labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask_image, connectivity=8)

    extractor = initialize_extractor()
    features = OrderedDict()

    for i in range(1, len(stats)):  # index 0 is the background component
        mask = np.uint8(mask_labels == i)
        result = extractor.execute(GetImageFromArray(image),
                                   GetImageFromArray(mask))
        if len(features) == 0:
            features['x'] = [int(centroids[i][0])]
            features['y'] = [int(centroids[i][1])]
            for key, val in six.iteritems(result):
                if key.startswith("original_"):
                    features[key] = [val]
        else:
            features['x'] += [int(centroids[i][0])]
            features['y'] += [int(centroids[i][1])]
            for key, val in six.iteritems(result):
                if key.startswith("original_"):
                    features[key] += [val]

    # POSSIBLE BUG, VERIFY THAT from_dict DOESN'T INFER WRONG TYPES
    # Note that we can return mask_labels here rather than an array of individual
    # masks - our sample order will match the labels in mask_labels
    return pd.DataFrame.from_dict(features).set_index(['x', 'y']), mask_labels

def get_mask_image(image, mask_params, verbosity=0):
    if type(mask_params) == str:
        with open(mask_params, 'r') as f:
            param_file = yaml.load(f, Loader=yaml.FullLoader)
            mode = param_file['mode']
            params = param_file['params']
    else:
        mode = mask_params['mode']
        params = mask_params['params']

    if verbosity:
        print("Using mode '%s'." % mode)
    
    if verbosity and params.get('clipLimit'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        hist1 = cv2.calcHist([image], [0], None, [256], [0,256])
        ax1.plot(hist1)
        ax1.set_title("Raw Image")
        ax1.set_xlim([0,256])
        ax1.set_ylim([0,hist1.max()])
        
    # Save a processed image for later
    # Note that these parameters are intentionally hard-coded
    # We are also deliberately not sharpening here
    if params.get('canny_unmasking'):
        enhanced = apply_contrast(image, 2)  # hard-coded factor
        canny = apply_canny(enhanced, *params.get('canny_unmasking'))
        canny_filled = apply_imfill(apply_closure(canny, 5))
        
    ### IMAGE PREPROCESSING ###
    
    if params.get('invert_image'):
        image = 255 - image

    if params.get('denoise_pre'):
        if verbosity:
            print("Denoising with intensity %i." % params['denoise_pre'])
        image = apply_denoise(image, params['denoise_pre'])

    if params.get('contrast'):
        if verbosity:
            print(("Enhancing contrast by factor of %.2f, with "
                   "offset=%s and preserve_background=%s.") %
                  (params['contrast'], params['offset'], params['preserve_background']))
        image = apply_contrast(image, params['contrast'],
                               params['offset'], params['preserve_background'])
    
    if params.get('clipLimit'):
        if verbosity:
            print("Equalizing histogram.")
        image = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8,8)) \
                   .apply(image)
    
    if params.get('denoise_post'):
        if verbosity:
            print("Denoising with intensity %i." % params['denoise_post'])
        image = apply_denoise(image, params['denoise_post'])

    if verbosity and params.get('clipLimit'):
        ax2.plot(cv2.calcHist([image], [0], None, [256], [0,256]))
        ax2.set_title("Preprocessed")
        ax2.set_xlim([0,256])
        ax2.set_ylim([0,hist1.max()])
    
    ### IMAGE BINARIZATION ###
        
    if mode == "background_filter":
        peak = cv2.calcHist([image], [0], None, [256], [0,256]).argmax()
        lower = peak - params['lower_width']
        upper = peak + params['upper_width']
        if verbosity:
            print("Filtering background between values %i and %i." % (lower, upper))
        binarized = np.uint8((image < lower) | (image > upper))
    elif mode == "edge_canny":
        if verbosity:
            print("Applying Canny with thresholds %i and %i." %
                  (params['canny_thresh1'], params['canny_thresh2']))
        image = apply_sharpen(image)
        binarized = apply_canny(image, params['canny_thresh1'], params['canny_thresh2'])
    elif mode == "edge_sobel":
        if verbosity:
            print("Applying Sobel with kernel size of %i." % params['sobel_ks'])
        image = apply_sharpen(image)
        sobel = apply_sobel(image, params['sobel_ks'])
        _, binarized = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif mode == "kmeans":
        if verbosity:
            print("Applying k-means with %i attempts." % params['attempts'])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(np.float32(raw_image.flatten()), 2, None, criteria,
                       params['attempts'], cv2.KMEANS_PP_CENTERS)
        binarized = np.uint8(labels.reshape(image.shape))
        if stats.mode(labels.flatten()).mode[0]:
            binarized = cv2.bitwise_not(binarized)  # POSSIBLE BUG, COMPARE TO (1 - binarized)
    elif mode == "thresh_adaptive":
        if verbosity:
            print("Applying adaptive thresholding with kernel size of %i and C=%i." %
                  (params['thresh_ks'], params['C']))
        binarized = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          params['thresh_ks'], params['C'])
    elif mode == "thresh_manual":
        if verbosity:
            print("Manually thresholding on values >=%i." % params['manual_threshold'])
        binarized = np.uint8(image >= params['manual_threshold'])
    elif mode == "thresh_otsu":
        if verbosity:
            print("Applying Otsu's.")
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        print("Invalid mask mode, aborting.")
        return None

    ### IMAGE POSTPROCESSING ###
    
    if params.get('border_removal'):
        if verbosity:
            print("Removing border of %i pixels." % params['border_removal'])
        binarized[:params['border_removal'], :] = 0
        binarized[:, :params['border_removal']] = 0
        binarized[-params['border_removal']:, :] = 0
        binarized[:, -params['border_removal']:] = 0
        
    if params.get('closure_ks'):
        if verbosity:
            print("Closing with kernel size of %i." % params['closure_ks'])
        filled = apply_imfill(apply_closure(binarized, kernel_size=params['closure_ks']))
    else:
        filled = apply_imfill(binarized)

    if params.get('canny_unmasking') or params.get('size_thresh'):
        if verbosity:
            if params.get('canny_unmasking'):
                print("Unmasking areas without edges using Canny%s:" % params.get('canny_unmasking'))
            if params.get('size_thresh'):
                print("Unmasking regions smaller than %i pixels." % params['size_thresh'])
        _, labelled, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        for i in range(1, len(stats)):  # index 0 is the background component
            if (params.get('canny_unmasking') and np.sum(np.uint8(labelled==i)*canny_filled) == 0) \
                or (params.get('size_thresh') and stats[i, -1] < params['size_thresh']):
                    filled = filled * (1 - np.uint8(labelled == i))

    if verbosity >= 2:
        display_image_array([image, binarized, filled],
                            ["Preprocessed", "Binarized", "Filled"], columns=3, dpi=450)

    return filled

def initialize_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # Enable everything but shape (3D) and glcm.SumAverage
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=[], shape2D=[],
                                   glcm=['Autocorrelation', 'JointAverage', 'ClusterProminence',
                                         'ClusterShade', 'ClusterTendency', 'Contrast',
                                         'Correlation', 'DifferenceAverage', 'DifferenceEntropy',
                                         'DifferenceVariance', 'JointEnergy', 'JointEntropy',
                                         'Imc1', 'Imc2', 'Idm', 'Idmn', 'Id', 'Idn',
                                         'InverseVariance', 'MaximumProbability', 'SumEntropy',
                                         'SumSquares'],
                                   glszm=[], glrlm=[], ngtdm=[], gldm=[])
    return extractor