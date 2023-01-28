import tensorflow as tf

def load_mnist_dataset():
    """
    Load the MNIST dataset using keras datasets
    X is the pixel data and Y is the number it represents
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)
