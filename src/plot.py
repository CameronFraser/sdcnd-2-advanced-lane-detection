import matplotlib.pyplot as plt
from math import ceil
import numpy as np

def plot_images(images, cols = 1, labels = None):
    n_images = len(images)
    if labels is None: labels = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, label) in enumerate(zip(images, labels)):
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(label, fontsize=20)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def plot_image(image, gray=False):
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
