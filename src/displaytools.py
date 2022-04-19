import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.manifold import TSNE

def display_classifications(image, centroids, labels, cmap=None):
    plt.figure(dpi=240)
    plt.axis('off')
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.scatter(*zip(*centroids), s=20, c=labels, marker='+', cmap=cmap, linewidths=0.75)
    plt.show()

def display_hist(image):
    plt.hist(image.ravel(), bins=255)
    plt.xlim([0,255])
    plt.show()
    plt.close()
            
def display_image(image, dpi=240):
    plt.figure(dpi=dpi)
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()

def display_image_array(im_arr, titles=None, columns=7, dpi=240):
    rows = int(len(im_arr) / columns) + 1
    plt.figure(dpi=dpi)
    for i in range(len(im_arr)):
        plt.subplot(rows,columns,i+1)
        plt.imshow(im_arr[i], interpolation='nearest', cmap='gray')
        if titles and len(im_arr) == len(titles):
            plt.title(titles[i], fontsize=5)
        plt.axis('off')
    plt.show()
    plt.close()

def display_tsne(df, labels):
    transformed = TSNE(learning_rate='auto', init='pca', random_state=0).fit_transform(df)
    plt.scatter(transformed[:,0], transformed[:,1], s=5, c=labels)
    plt.title("t-SNE")
    plt.show()

def draw_contours(image, mask, color='red', filled=False):
    color = np.array(matplotlib.colors.to_rgb(color))*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = -1 if filled else 1
    return cv2.drawContours(image, contours, contourIdx=-1, color=color, thickness=thickness)

def get_contoured_image(image, mask_labels, labels, colormap, filled=False):
    contours = np.full(mask_labels.shape, -1, dtype=int)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in range(len(labels)):
        contours[mask_labels==i+1] = labels[i]
    for i in range(np.max(contours)+1):
        image = draw_contours(image, np.uint8(contours==i), colormap.colors[i])
    return image

def print_label_counts(labels, colormap=None):
    c = Counter(labels)
    for k in sorted(c):
        if colormap:
            print("Class %s (%s): %i (%.2f%%)" %
                  (k, colormap.colors[k], c[k], c[k]*100/sum(c.values())))
        else:
            print("Class %s: %i (%.2f%%)" % (k, c[k], c[k]*100/sum(c.values())))
