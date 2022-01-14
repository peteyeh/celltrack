import matplotlib.pyplot as plt

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
