import random
import numpy as np
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
        biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]

        #weights = []
        #for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
        #    weights.append(np.random.randn(x,y)/np.sqrt(x))
        weights = [np.random.randn(y, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        return weights, biases

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate):
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
            print(f"Epoch {i} completed")

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

    def backpropagation(self, picture, number):
        """
        This will be the backrpopagation algorithm
        """
        change_to_bias = [np.zeros(b.shape) for b in self.biases]
        change_to_weight = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activations, z_vectors = self.feedforward(picture, number)
        
        # Calculate the gradient of the cost function
        error = self.cost_derivative(activations[-1], number) * self.sigmoid_prime(z_vectors[-1])

        # Modify the weights and biases in connections to the output layer
        change_to_bias[-1] = error
        change_to_weight[-1] = np.dot(error, activations[-2].T)

        for i in range(2, self.number_of_layers):
            z_vector = z_vectors[-i]
            sigmoid_prime = self.sigmoid_prime(z_vector)
            error = np.dot(self.weights[-i+1].T, error) + sigmoid_prime
            change_to_bias[-i] = error
            change_to_weight[-i] = np.dot(error, activations[-i-1].T)

        return (change_to_bias, change_to_weight)

    def feedforward(self, picture, number):
        """
        Feedforward algorithm, computes the output of the
        neural network for a given input AKA makes a prediction
        of what the given input is
        """
        activation = picture

        # Store all the activations to a list
        activations = [picture]

        # Store all the z vectors to a list
        z_vectors = []

        biases_and_weights = zip(self.biases, self.weights)
        for bias, weight in biases_and_weights:
            z_vector = np.dot(weight, activation) + bias
            z_vectors.append(z_vector)

            # Apply the activation function
            activation = self.sigmoid(z_vector)
            activations.append(activation)

        return activations, z_vectors

    def cost_derivative(self, output_layer_activations, target_values):
        """
        Gradient of the cost function, calculates the difference
        between the predicted output and target values
        """
        return (output_layer_activations - target_values)

    def sigmoid(self, z_vector):
        """
        Sigmoid activation function
        """
        return (1.0/(1.0 + np.exp(-z_vector)))

    def sigmoid_prime(self, z_vector):
        """
        Sigmoid functions derivative
        """
        return self.sigmoid(z_vector)*(1-self.sigmoid(z_vector))


# Test NN
nn = NeuralNetwork([784, 16, 10])
(x_train, y_train), (x_test, y_test) = load_mnist_dataset()
training_data = list(zip(x_train, y_train))
nn.stochastic_gradient_descent(list(training_data), 5, 30, 3.0)
