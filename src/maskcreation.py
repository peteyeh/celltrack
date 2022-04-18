import cv2
import numpy as np
import yaml

from scipy import stats

if __name__ == "__main__":
    import imageio, os, shutil, sys
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
        canny_filled = np.uint8(apply_imfill(apply_closure(canny, 5)) != 0)
        
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

# Note that we cannot simply return mask_image here because we don't apply closure
# when testing total_area.
def refine_offset(image, mask_params=default_params, left=5, right=5):
    if type(mask_params) == str:
        with open(mask_params, 'r') as f:
            param_file = yaml.load(f, Loader=yaml.FullLoader)
            mode = param_file['mode']
            base_params = param_file['params']
    else:
        mode = mask_params['mode']
        base_params = mask_params['params']

    params = base_params.copy()
    if params.get('closure_ks'):
        del params['closure_ks']

    search_range = range(base_params['offset']-left, base_params['offset']+right+1)

    total_area = []
    for offset in search_range:
        params['offset'] = offset
        mask_image = get_mask_image(image, {'mode': mode, 'params': params})
        total_area += [np.sum((mask_image != 0).astype(int)),]

    base_params['offset'] = search_range[np.where(np.diff(total_area) > 10000)[0][0]]
    return {'mode': mode, 'params': base_params}

def get_mask_image_with_refined_offset(image, mask_params=default_params, left=5, right=5):
    return get_mask_image(image, refine_offset(image, mask_params, left, right))

if __name__ == "__main__":
    import imageio, os, shutil, sys
    from datetime import datetime
    from joblib import cpu_count, delayed, Parallel
    from tqdm import tqdm

    from displaytools import *
    from improcessing import *

    try:
        image_stack = cv2.imreadmulti(sys.argv[1], flags=cv2.IMREAD_GRAYSCALE)[1]
        if len(image_stack) == 0:
            raise Exception()
    except:
        print("Unable to read image stack. Make sure you execute with:")
        print("  python3 maskcreation.py image_path [output_path]")
        sys.exit(1)

    print("Creating %i mask images:" % len(image_stack))
    mask_images = \
        Parallel(n_jobs=cpu_count())(
            delayed(get_mask_image_with_refined_offset)(_) for _ in tqdm(image_stack))

    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist
    base_path = os.path.join(out_path, os.path.basename(sys.argv[1]).split('.')[0])
    if not os.path.exists(base_path):
        print("Creating base directory %s." % base_path)
        os.mkdir(base_path)
    write_path = os.path.join(base_path, datetime.now().strftime("%Y%m%d%H%M%S"), "mask_images")
    print("Writing mask images to " + write_path + ":")
    os.mkdir(os.path.split(write_path)[0])
    os.mkdir(write_path)
    for i in tqdm(range(len(mask_images))):
        imageio.imwrite((write_path + "/%i.png" % i), mask_images[i])
