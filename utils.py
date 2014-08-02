import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def read_images(filepath):
    with open(filepath, 'rb') as _file:
        dtype = np.uint8
        magic, size, rows, cols = struct.unpack('>IIII', _file.read(16))
        if magic != 2051:
            raise ValueError('Got wrong magic number. {0}'.format(magic))
        images = np.fromfile(_file, dtype=dtype)
        images = np.reshape(images, (size, rows*cols))
    return (size, rows, cols, images)


def read_labels(filepath):
    with open(filepath, 'rb') as _file:
        dtype = np.uint8
        magic, size = struct.unpack('>II', _file.read(8))
        if magic != 2049:
            raise ValueError('Got wrong magic number.')
        labels = np.fromfile(_file, dtype=dtype)
    return (size, labels)


def display(image, width=28, height=28):
    image = np.reshape(image, [width, height])
    plt.imshow(image, cmap=cm.binary)
    plt.show()


def visualize_thetas(thetas):
    fig, axes = plt.subplots(nrows=3, ncols=4)
    axes = axes.flatten()
    for i in range(10):
        image = np.reshape(thetas[1:, i], [28, 28])
        axes[i].imshow(image, cmap=plt.cm.binary)
    plt.show()
