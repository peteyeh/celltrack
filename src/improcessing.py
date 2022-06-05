import cv2
import numpy as np

from scipy import stats

def apply_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size,kernel_size), cv2.BORDER_DEFAULT)

def apply_canny(image, threshold1=130, threshold2=180):
    return cv2.Canny(np.uint8(image), threshold1=threshold1, threshold2=threshold2)

def apply_closure(image, kernel_size=5):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# preserve_background will restore the background to its original value;
# otherwise we will clip anything darker than the background
def apply_contrast(image, factor=2, peak_offset=0, preserve_background=True):
    peak = cv2.calcHist([image], [0], None, [256], [0,256]).argmax()
    image = np.float64(image) - peak
    if peak_offset:
        image += peak_offset
    image *= factor
    if preserve_background:
        image = image + peak - peak_offset
    return np.uint8(np.clip(image, 0, 255))

def apply_denoise(image, h=3):
    return cv2.fastNlMeansDenoising(image, h=h)

def apply_dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.dilate(image, kernel)

def apply_imfill(image):
    t = image.copy()
    y_dim, x_dim = image.shape
    mask = np.zeros((y_dim+2,x_dim+2), np.uint8)
    t = cv2.floodFill(t, mask, (0,0), 1)[1]
    t = cv2.floodFill(t, mask, (0, int(y_dim/2)), 1)[1]
    t = cv2.floodFill(t, mask, (0, y_dim-1), 1)[1]
    t = cv2.floodFill(t, mask, (int(x_dim/2), 0), 1)[1]
    t = cv2.floodFill(t, mask, (int(x_dim/2), y_dim-1), 1)[1]
    t = cv2.floodFill(t, mask, (x_dim-1, 0), 1)[1]
    t = cv2.floodFill(t, mask, (x_dim-1, int(y_dim/2)), 1)[1]
    t = cv2.floodFill(t, mask, (x_dim-1, y_dim-1), 1)[1]
    return image | (t == 0)  # we can intermingle bits and booleans here, (1 | False) == 1

def apply_sharpen(image):
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    # ddepth=-1 preserves the source depth
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def apply_sobel(image, kernel_size=5):
    # ddepth=-1 preserves the source depth
    return cv2.Sobel(src=image, ddepth=-1, dx=1, dy=1, ksize=kernel_size)

def binarize_image(image):
    return np.uint8(image != 0)

# Returns the first mode in the image
def get_mode(image):
    return stats.mode(image, axis=None)[0][0]

# connectivity: 4 to exclude diagonals, 8 to include
def get_num_components(image, connectivity=8):
    return cv2.connectedComponentsWithStats(image, connectivity)[0] - 1

def scale_image(image, max_value=255, mode=90):
    image = np.float64(image)
    if mode:
        image -= get_mode(image)

        # Stretch distribution in both directions
        target_max = max_value - mode
        target_min = 0 - mode
        image[image > 0] *= (target_max / image.max())
        if image.min() != 0: # this avoids a divide-by-zero warning
            image[image < 0] *= (target_min / image.min())

        image += mode
        image = np.around(image)  # Need to remove rounding error first
        # We're returning in uint8, so it's incredibly important to check here
        if image.min() < 0:
            raise Exception("Min value is %d." % image.min())
        if image.max() > max_value:
            raise Exception("Max value is %d." % image.max())
        return np.uint8(image)
    else:
        image -= image.min()
        return np.uint8(image * max_value / image.max())
