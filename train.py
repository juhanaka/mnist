from __future__ import division
import numpy as np
import functions as func
import utils
import matplotlib.pyplot as plt

# Only using the training dataset for this.
TRAINING_IMAGES_PATH = './training_data/train-images-idx3-ubyte'
TRAINING_LABELS_PATH = './training_data/train-labels-idx1-ubyte'

N_LABELS = 10

M_TRAINING = 5000
M_TESTING = 1000


def train(alpha, n_iter):
    size_img, rows, cols, images = utils.read_images(
        TRAINING_IMAGES_PATH
    )
    size_lbl, labels = utils.read_labels(TRAINING_LABELS_PATH)

    images = images[range(0, M_TRAINING), :]
    labels = labels[range(0, M_TRAINING), :]
    size_img = M_TRAINING

    bias_terms = np.ones([size_img, 1], dtype=np.float64)
    images = np.concatenate((bias_terms, images), axis=1).astype(np.float64)
    thetas = np.zeros([rows*cols+1, N_LABELS], dtype=np.float64)
    costs = np.zeros([n_iter, N_LABELS])
    X = images / 255
    for i in range(N_LABELS):
        print 'Training a classifier for label {0}'.format(i)
        y = np.array([[1 if label == i else 0 for label in labels]]).T
        thetas[:, i:i+1], costs[:, i:i+1] = func.gradient_descent(
            thetas[:, i:i+1],
            y, X, alpha,
            n_iter
        )
        plt.plot(costs[:, i:i+1])
        plt.show()
    return thetas


def test(thetas):
    size_img, rows, cols, images = utils.read_images(
        TRAINING_IMAGES_PATH
    )
    size_lbl, labels = utils.read_labels(TRAINING_LABELS_PATH)

    bias_terms = np.ones([size_img, 1])
    images = np.concatenate((bias_terms, images), axis=1)

    training_images = images[range(0, M_TRAINING), :]
    training_labels = labels[range(0, M_TRAINING), :]

    X = training_images / 255
    y = training_labels
    accuracy = _test(thetas, y, X, M_TRAINING)

    print 'Trained with {0} examples. Training accuracy: {1}'.format(
        M_TRAINING,
        accuracy
    )

    test_labels = labels[range(M_TRAINING, M_TRAINING + M_TESTING), :]
    test_images = images[range(M_TRAINING, M_TRAINING + M_TESTING), :]

    X = test_images / 255
    y = test_labels
    accuracy = _test(thetas, y, X, M_TESTING)

    print 'Tested with {0} examples. Test accuracy: {1}'.format(M_TESTING,
                                                                accuracy)


def _test(thetas, y, X, size_img):
    h = np.zeros([size_img, N_LABELS])
    for i in range(N_LABELS):
        h[:, i] = func.hypothesis(thetas[:, i], X)
    final_h = np.transpose(np.argmax(h, 1))
    truth = np.equal(final_h, y)
    return np.sum(truth) / size_img


def predict(thetas, image):
    h = np.zeros([1, N_LABELS])
    bias_term = np.ones([1, ])
    image = np.concatenate((bias_term, image), axis=1)
    for i in range(N_LABELS):
        h[:, i] = func.hypothesis(thetas[:, i], image)
    return np.argmax(h, 1)[0]
