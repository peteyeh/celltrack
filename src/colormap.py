import matplotlib.colors

colormap = matplotlib.colors.ListedColormap(['yellow', 'red', 'cyan', 'orange', 'green',
                                             'purple', 'blue', 'magenta', 'lime', 'dodgerblue'])

def get_colormap():
    return colormap

def get_color(i):
    return colormap.colors[i]