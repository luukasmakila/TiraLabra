# Week 2

## What has been going on?
- Started the coding of the NeuralNetwork class. Currently the NeuralNetwork can be initialized and given training data is split up to mini-batches.
- Added a load_mnist_dataset function to /helpers directory. Currently it pulls the data using tf.keras.datasets. Later if there's time left I will change this to using my own way of handling the data.
- Added a /tests directory for tests. There is not much to test yet since the methods rely heavily on eachother and they are not implemented fully yet.

## Plans for next week?
- Complete the main beef of the Neural Network so complete the stochastic_gradient_descent, backpropagation and update_weights_and_biases methods. These will be tweaked for better performance as the project advances.
