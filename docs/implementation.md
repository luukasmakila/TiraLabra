# Digit recognition with a neural network

## Purpose

Learn how to teach a computer to recognize handwritten digits in pictures using a neural network.

## Neural Network

The neural network is built from scratch with Python and Numpy. MNIST dataset is used for training and testing data. Networks architecture can be modified by the user, meaning the user can experiment with different layer configurations and hyperparameters and see how that inpacts the training of the model. At the end of training the network user can visualize the predictions of the neural network against the actual labels of the images.

Methods used to achieve the end goal include: backpropagation, forwardpropagation, onehot encoding the labels, sigmoid as the activation function and mini-batch gradient descent.

Time complexity is dependent of the number of layers, the number of neurons in these layers, size of the training dataset aswell as the number of epochs. For the setup I used for the demo the time complexity is <b>O(nt\*(ij+2j+jk))</b>, where <b>n</b> is the number of epochs (training rounds), <b>t</b> is the number of training examples and <b>i, j and k</b> represent the amount of neurons in each of the layers in the network.

The MNIST data is processed to be in the correct format before training. Images are scaled down meaning the the color is between 0-1 instead of 0-255. After that the images are reshaped to be 784 by 1 matrixes. The labeled data is onehot encoded and then the data is ready to be fed to the network.
