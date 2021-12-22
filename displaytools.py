import matplotlib.pyplot as plt
            
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
