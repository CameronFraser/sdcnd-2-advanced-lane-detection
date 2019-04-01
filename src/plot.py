import matplotlib.pyplot as plt
from math import ceil
import numpy as np

def plot_images(images, columns, gray=False):
    try:
        height, width = images[0].shape
    except:
        height, width, _ = images[0].shape
    
    fig_size = len(images) * 2

    fig = plt.figure(figsize=(fig_size, fig_size))
    rows = ceil(len(images)/float(columns))
    
    for i in range(len(images)):
        fig.add_subplot(rows, columns, i+1)
        if gray:
            plt.imshow(images[i], cmap="gray")
        else:
            plt.imshow(images[i])
    plt.show()

def plot_image(image):
    plt.clf()
    plt.imshow(image)
    plt.show()