import matplotlib.image as mpimg

def load_image(file_name):
    return mpimg.imread(file_name)

def load_images(file_names):
    images = []
    for file_name in file_names:
        images.append(mpimg.imread(file_name))
    return images