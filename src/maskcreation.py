import cv2
import numpy as np
import yaml

from scipy import stats

if __name__ == "__main__":
    import imageio, os, shutil, sys
    from datetime import datetime
    from joblib import cpu_count, delayed, Parallel
    from tqdm import tqdm

    from displaytools import *
    from improcessing import *
    default_params = 'params_gray_mask.yml'

else:  # kinda hacky but it works
    from src.displaytools import *
    from src.improcessing import *
    default_params = 'src/params_gray_mask.yml'

def get_mask_image(image, mask_params=default_params, verbosity=0):
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
        import matplotlib.pyplot as plt  # this is an expensive import, so put it here
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
        # Binarize this so that we can apply operations on it later
        canny_filled = binarize_image(apply_imfill(apply_closure(canny, 5)))
        
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
        binarized = binarize_image(apply_canny(image, params['canny_thresh1'], params['canny_thresh2']))
    elif mode == "edge_sobel":
        if verbosity:
            print("Applying Sobel with kernel size of %i." % params['sobel_ks'])
        image = apply_sharpen(image)
        sobel = apply_sobel(image, params['sobel_ks'])
        binarized = binarize_image(cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    elif mode == "kmeans":
        if verbosity:
            print("Applying k-means with %i attempts." % params['attempts'])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(np.float32(image.flatten()), 2, None, criteria,
                       params['attempts'], cv2.KMEANS_PP_CENTERS)
        binarized = np.uint8(labels.reshape(image.shape))
        if stats.mode(labels.flatten()).mode[0]:
            binarized = np.uint8(binarized == 0)
    elif mode == "thresh_adaptive":
        if verbosity:
            print("Applying adaptive thresholding with kernel size of %i and C=%i." %
                  (params['thresh_ks'], params['C']))
        binarized = binarize_image(cv2.adaptiveThreshold(image, 255,
                                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV,
                                                         params['thresh_ks'], params['C']))
    elif mode == "thresh_manual":
        if verbosity:
            print("Manually thresholding on values >=%i." % params['manual_threshold'])
        binarized = np.uint8(image >= params['manual_threshold'])
    elif mode == "thresh_otsu":
        if verbosity:
            print("Applying Otsu's.")
        binarized = binarize_image(cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1])

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

    # Unmask using size first because it is quicker
    if params.get('size_thresh'):
        if verbosity:
            print("Unmasking regions smaller than %i pixels." % params['size_thresh'])
        _, labelled, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        filled = np.uint8(np.isin(labelled, np.where(stats[1:,-1] >= params['size_thresh'])[0]+1))

    if params.get('canny_unmasking'):
        if verbosity:
            print("Unmasking areas without edges using Canny%s:" % params.get('canny_unmasking'))
        _, labelled, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        for i in range(1, len(stats)):  # index 0 is the background component
            if not np.any(canny_filled[labelled==i]):
                filled[labelled==i] = 0

    if verbosity >= 2:
        display_image_array([image, binarized, filled],
                            ["Preprocessed", "Binarized", "Filled"], columns=3, dpi=450)

    return filled

def get_mask_image_with_refined_offset(image, mask_params=default_params, left=20, right=5, verbosity=0):
    if type(mask_params) == str:
        with open(mask_params, 'r') as f:
            param_file = yaml.load(f, Loader=yaml.FullLoader)
            mode = param_file['mode']
            params = param_file['params']
    else:
        mode = mask_params['mode']
        params = mask_params['params']

    search_range = range(params['offset']-left, params['offset']+right+1)

    if verbosity:
        print("Searching through range %s." % str(search_range))

    prev_true = None
    prev_test = None
    for offset in search_range:
        params['offset'] = offset  # keep advancing params['offset'] and roll back if we go too far
        true_mask = get_mask_image(image, {'mode': mode, 'params': params}, verbosity)
        if not validate_mask(true_mask):
            if prev_true is not None:
                return prev_true
            continue

        test_params = params.copy()
        test_params['closure_ks'] = 5
        del test_params['size_thresh']
        test_mask = get_mask_image(image, {'mode': mode, 'params': test_params}, verbosity)
        if not validate_mask(test_mask, cutoff=0.4):
            if prev_true is not None:
                return prev_true
            continue

        if prev_true is not None and prev_test is not None:
            true_diff = true_mask - prev_true
            test_diff = test_mask - prev_test
            max_true_diff = np.max(cv2.connectedComponentsWithStats(true_diff, connectivity=8)[2][1:,-1])
            max_test_diff = np.max(cv2.connectedComponentsWithStats(test_diff, connectivity=8)[2][1:,-1])
            # could also check for num_components > 1 instead of np.any for speed, but this is cleaner
            if np.any(true_diff) and np.any(test_diff) and \
               (max_true_diff > 10000 or (max_true_diff > 2000 and max_test_diff > 5000)):
                return prev_true

        prev_true = true_mask
        prev_test = test_mask
    return true_mask

def validate_mask(mask_image, cutoff=0.15):
    # if nothing is masked or if everything is masked
    if not np.any(mask_image) or np.all(mask_image):
        return False
    largest_area = np.max(cv2.connectedComponentsWithStats(mask_image, connectivity=8)[2][1:,-1])
    image_area = mask_image.shape[0] * mask_image.shape[1]
    if largest_area > cutoff * image_area:
        return False
    return True

if __name__ == "__main__":
    try:
        image_stack = list(map(scale_image, cv2.imreadmulti(sys.argv[1], flags=cv2.IMREAD_GRAYSCALE)[1]))
        if len(image_stack) == 0:
            raise Exception()
    except:
        print("Unable to read image stack. Make sure you execute with:")
        print("  python3 maskcreation.py image_path [output_path]")
        sys.exit(1)

    basename = os.path.basename(sys.argv[1]).split('.')[0]
    print("Creating %i mask images for %s:" % (len(image_stack), basename))
    mask_images = \
        Parallel(n_jobs=cpu_count())(
            delayed(get_mask_image_with_refined_offset)(_) for _ in tqdm(image_stack))

    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist
    base_path = os.path.join(out_path, basename)
    if not os.path.exists(base_path):
        print("Creating base directory %s." % base_path)
        os.mkdir(base_path)
    write_path = os.path.join(base_path, datetime.now().strftime("%Y%m%d%H%M%S"), "mask_images")
    print("Writing mask images to " + write_path + ":")
    os.mkdir(os.path.split(write_path)[0])
    os.mkdir(write_path)
    for i in tqdm(range(len(mask_images))):
        imageio.imwrite((write_path + "/%i.png" % i), mask_images[i])
