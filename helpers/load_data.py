import numpy as np
import tensorflow as tf

def load_mnist_dataset():
    """
    Load the MNIST dataset using keras datasets
    X is the pixel data and Y is the number it represents
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scale down the image data to 0-1
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Vecotrize the input data
    x_train = [np.reshape(i, (784, 1)) for i in x_train]
    x_test = [np.reshape(i, (784, 1)) for i in x_test]

    # Vecotrize the labeld data
    y_train = [vectorize_result(i) for i in y_train]
    y_test = [vectorize_result(i) for i in y_test]

    return (x_train, y_train), (x_test, y_test)

def vectorize_result(i):
    """
    Returns a 10-dimensional unitvector
    representing the labeled data
    """
    # Make the i:th position one and the rest zeroes
    vector = np.zeros((10, 1))
    vector[i] = 1
    return vector
