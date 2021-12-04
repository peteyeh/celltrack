import cv2
import numpy as np

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

def apply_impipeline(image):
    return apply_imfill(apply_closure(apply_canny(image)))

def get_box(image, x, y, box_x, box_y):
    # box should not exceed bounds of image
    return image[max(y-box_y, 0):min(y+box_y, image.shape[0]),
                 max(x-box_x, 0):min(x+box_x, image.shape[1])]

# connectivity: 4 to exclude diagonals, 8 to include
def get_num_components(image, connectivity=8):
    return cv2.connectedComponentsWithStats(image, connectivity)[0] - 1

# Input:
#  - image: binary image
#  - threshold [0-1]: only remove components less that make up less than threshold
#                     amount of total components
#  - connectivity: 4 to exclude diagonals, 8 to include
def remove_secondary_components(image, threshold=0.4, connectivity=8):
    imtype = image.dtype
    num_components, labelled, stats, centroids = \
        cv2.connectedComponentsWithStats(image, connectivity)
    
    # There is nothing in the image (the whole image is the component)
    if num_components == 1:
        return image
    
    component_sizes = stats[1:,-1]  # label 0 is the background component
    total_size = sum(component_sizes)
    largest_component = np.argmax(component_sizes) + 1

    for c in range(1, num_components):  # label 0 is the background component
        if c == largest_component:
            continue
        component = (labelled == c).astype(int)
        if np.sum(component) < threshold*total_size:
            image = image & (1-component)
    
    return image.astype(imtype)
