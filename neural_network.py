import random
import numpy as np
import matplotlib.pyplot as plt
from helpers.load_data import load_mnist_dataset

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights, self.biases = self.init_weights_and_biases()
    
    def init_weights_and_biases(self):
        """
        Initialize weights and biases, ignore input layer
        """
        biases = [np.random.rand(y, 1) - 0.5 for y in self.layer_sizes[1:]]
        weights = [np.random.normal(scale=0.5, size=(y, x)) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        return weights, biases

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
        """
        This is main part in making the neural network learn
        Uses mini-batch stochastic gradient descent for
        faster computation

        Epochs is the number of training rounds

        Learning rate determines the size of the "step"
        we take when moving towards the minimum loss
        aka how fast or slow we move towards the optimal
        weights and biases
        """
        if test_data:
            test_data_length = len(test_data)

        training_data_length = len(training_data)
        for i in range(epochs):
            # Divide the training data into random mini-batches
            random.shuffle(training_data)
            mini_batches = [
                training_data[j:j + mini_batch_size]
                for j in range(0, training_data_length, mini_batch_size)
            ]
            for bacth in mini_batches:
                self.update_weights_and_biases(bacth, learning_rate)
            print(f"Epoch {i}")
            if test_data:
                print(f"{self.evaluate_model(test_data)} / {test_data_length}")

    def update_weights_and_biases(self, batch, learning_rate):
        """
        This updates the NN's weights and biases
        by computing the gradient for the given
        mini-batch
        """

        # Initialize 2 lists to store the changes to the weights and biases
        total_change_bias = [np.zeros(b.shape) for b in self.biases]
        total_change_weight = [np.zeros(w.shape) for w in self.weights]

        for picture, number in batch:
            # Calculate the change in weight and bias for a single training example with backrpopagation
            change_to_bias, change_to_weight = self.backpropagation(picture, number)
            total_change_bias = [tcb + ctb for tcb, ctb in zip(total_change_bias, change_to_bias)]
            total_change_weight = [tcw + ctw for tcw, ctw in zip(total_change_weight, change_to_weight)]

        # Modify the weights and biases
        self.biases = [b - (learning_rate/len(batch)) * tcb for b, tcb in zip(self.biases, total_change_bias)]
        self.weights = [w - (learning_rate/len(batch)) * tcw for w, tcw in zip(self.weights, total_change_weight)]

    def forwardpropagation(self, picture):
        """
        This is the forwardpropagation algorithm
        """
        activation = picture
        activations = [picture]

        z_vectors = []

        for weight, bias in zip(self.weights, self.biases):
            z_vector = np.dot(weight, activation) + bias
            z_vectors.append(z_vector)

            activation = sigmoid(z_vector)
            activations.append(activation)

        return activations, z_vectors

    def backpropagation(self, picture, number):
        """
        This the backpropagation algorithm
        """
        change_to_bias = [np.zeros(b.shape) for b in self.biases]
        change_to_weight = [np.zeros(w.shape) for w in self.weights]

        activations, z_vectors = self.forwardpropagation(picture)

        # Calculate the gradient of the cost function
        error = self.cost_derivative(activations[-1], number) * sigmoid_prime(z_vectors[-1])

        # Modify the weights and biases in connections to the output layer
        change_to_bias[-1] = error
        change_to_weight[-1] = np.dot(error, activations[-2].T)

        for i in range(2, self.number_of_layers):
            z_vector = z_vectors[-i]
            sp = sigmoid_prime(z_vector)
            error = np.dot(self.weights[-i+1].T, error) * sp
            change_to_bias[-i] = error
            change_to_weight[-i] = np.dot(error, activations[-i-1].T)

        return (change_to_bias, change_to_weight)

    def feedforward(self, x):
        """
        Returns the networks output for a given input
        """
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x

    def evaluate_model(self, test_data):
        """
        Get the number of test inputs that the neural
        network recognizes correctly
        """
        test_results = []
        for (x, y) in test_data:
            test_results.append((np.argmax(self.feedforward(x)), np.argmax(y)))
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_layer_activations, target_values):
        """
        Gradient of the cost function, calculates the difference
        between the predicted output and target values
        """
        return (output_layer_activations - target_values)

    def visualize(self, data):
        """
        Method to visualize the predictions made by the neural network
        """

        for x, y in data:
            activations = self.feedforward(x)
            predictions = activations

            #if np.argmax(predictions) != np.argmax(y):
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.title(f'Prediction: {np.argmax(predictions)}, Real label: {np.argmax(y)}')

            plt.show()
            # Display the whole vector of predictions
            print(f'Prediction vector: {predictions}')


# Activation functions
def sigmoid(z_vector):
    """
    Sigmoid activation function
    """
    return (1.0/(1.0 + np.exp(-z_vector)))

def sigmoid_prime(z_vector):
    """
    Sigmoid functions derivative
    """
    return sigmoid(z_vector)*(1-sigmoid(z_vector))


# Test NN
nn = NeuralNetwork([784, 100, 10])
(x_train, y_train), (x_test, y_test) = load_mnist_dataset()
training_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))
nn.stochastic_gradient_descent(list(training_data), 10, 10, 3.0, list(test_data))
nn.visualize(list(test_data))
