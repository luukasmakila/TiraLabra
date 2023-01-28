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
        biases = [np.random.randn(i, 1) for i in self.layer_sizes[1:]]

        weights = []
        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weights.append(np.random.randn(x,y)/np.sqrt(x))

        return weights, biases

nn = NeuralNetwork([784,16,16,10])
