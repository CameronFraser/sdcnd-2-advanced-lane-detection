import matplotlib.pyplot as plt
from math import ceil
import numpy as np

def plot_images(images):
    height, width, _ = images[0].shape
    fig = plt.figure(figsize=(100, 100))
    columns = 4
    rows = ceil(len(images)/float(columns))
    for i in range(len(images)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(images[i])
    plt.show()