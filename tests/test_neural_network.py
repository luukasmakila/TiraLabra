import pytest
from neural_network import NeuralNetwork

def test_init_neural_network():
    """
    Test initializing the Neural Network class
    """
    nn = NeuralNetwork([784, 16, 16, 10])
    assert len(nn.weights) == 3
    assert len(nn.biases) == 3
