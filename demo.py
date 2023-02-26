from neural_network import NeuralNetwork
from helpers.load_data import load_mnist_dataset

# Input layer needs to be of size 784 and output layer of the size 10
# Hidden layers in between can be modified to get different results
architecture = [784, 100, 10]

# Number of training rounds
epochs = 5

# Size of the mini-batches
mini_batch_size = 10

# Learning rate
learning_rate = 3.0

# Test NN
nn = NeuralNetwork(architecture)

(x_train, y_train), (x_test, y_test) = load_mnist_dataset()

training_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))

nn.stochastic_gradient_descent(training_data=training_data,
                               epochs=epochs,
                               mini_batch_size=mini_batch_size,
                               learning_rate=learning_rate,
                               test_data=test_data)

nn.visualize(test_data)