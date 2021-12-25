import cv2
import numpy as np

def apply_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size,kernel_size), cv2.BORDER_DEFAULT)

def apply_canny(image, threshold1=100, threshold2=230):
    return cv2.Canny(np.uint8(image), threshold1=threshold1, threshold2=threshold2)

def apply_closure(image, kernel_size=5):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_imfill(image):
    t = image.copy()
    y_dim, x_dim = image.shape
    mask = np.zeros((y_dim+2,x_dim+2), np.uint8)
    t = cv2.floodFill(t, mask, (0,0), 255)[1]
    t = cv2.floodFill(t, mask, (0, y_dim-1), 255)[1]
    t = cv2.floodFill(t, mask, (x_dim-1, 0), 255)[1]
    t = cv2.floodFill(t, mask, (x_dim-1, y_dim-1), 255)[1]
    return image | cv2.bitwise_not(t)

# connectivity: 4 to exclude diagonals, 8 to include
def get_num_components(image, connectivity=8):
    return cv2.connectedComponentsWithStats(image, connectivity)[0] - 1
