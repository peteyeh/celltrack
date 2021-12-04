# Presumes that cells is organized as follows:
# [[x, y, box_x, box_y, merge_events, distance_travelled], ...]

import cv2
import matplotlib.pyplot as plt
import numpy as np

# gif_mode is needed as imageio uses RGB, whereas cv2 uses BGR
def add_centroids_and_boxes_to_image(image, cells, gif_mode=False):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in range(len(cells)):
        cell = cells[i]
        merge_events = cell[4]
        if merge_events == -1:
            continue
        elif merge_events >= 3:  # GREEN
            color = (0, 255, 0)
        elif merge_events >= 1:  # BLUE
            color = (0, 0, 255) if gif_mode else (255, 0, 0)
        else:                    # RED
            color = (255, 0, 0) if gif_mode else (0, 0, 255)
        image = cv2.circle(image, cell[0:2], 0, color, 8)
        r_xdim = (cell[0]-cell[2], cell[1]-cell[3])
        r_ydim = (cell[0]+cell[2], cell[1]+cell[3])
        image = cv2.rectangle(image, r_xdim, r_ydim, color, 2)
    return image

def display_centroids(image, cells, dpi=150, save_name=None, mode='cv2'):
    if mode=='matplotlib':
        plt.figure(dpi=dpi)
        plt.imshow(image, interpolation='nearest', cmap='gray')
        overlap_idx = np.where(cells[:,4] == 1)[0]
        merged_idx = np.where(cells[:,4] == -1)[0]
        if len(overlap_idx) > 0:
            plt.scatter(*zip(*cells[overlap_idx,0:2]), c='lime', marker='+')
        r_cells = np.delete(cells, np.append(overlap_idx, merged_idx), axis=0)
        plt.scatter(*zip(*r_cells[:,0:2]), c='r', marker='+')
        if save_name is not None:
            plt.savefig('saved_images/' + save_name + '.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    elif mode=='cv2':
        image = add_centroids_and_boxes_to_image(image, cells)
        if save_name is not None:
            cv2.imwrite('saved_images/' + save_name + '.png', image)
        else:
            plt.figure(dpi=dpi)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
def display_image(image, dpi=240):
    plt.imshow(image, interpolation='nearest', cmap='gray')

def display_image_array(im_arr, columns=7, dpi=240):
    rows = int(len(im_arr) / columns) + 1
    plt.figure(dpi=dpi)
    for i in range(len(im_arr)):
        plt.subplot(rows,columns,i+1)
        plt.imshow(im_arr[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.show()
    plt.close()
