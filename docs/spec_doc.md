# Project specification

In this documentation we'll go over the details of the project and how it will be implemented. To understand it better i've explained the architecture and some terminology below aswell.

## What is it?
The goal of the project is to make and train a fully functioning neural network to recognize handwritten digits. This is apparently the classic starter project in ML and im hoping that at the end of the project I will have a decent understanding of neural networks and the different algorithms behind training them.

## How will it be done?
To maximize learning the project will be created from scratch using Python and Numpys. Tensorflow etc. will not be used for the model creation and training since it would literally be like 5 lines of code. Model will be trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset consisting of labeled pictures of handwritten digits.

## The architecture of the neural network
You can think of the neural network as a function. You give it <b>x</b> and it spits out <b>y</b>.

The neural network will be structured like the following:
- The input layer. This is where we take in the data that we want to recognize. In the case of handwritten digits and the MNIST dataset the pictures are 28x28 pixels. Knowing this the input layer will be a "flattened line" of the picture so it's a layer consisting of 784 neurons (28 * 28), each neuron representing one pixel of the picture. These pixels will be scaled down to between 0 and 1 on the greyscale, meaning that each pixel represents only an amount of light. This number between 0 and 1 will then determine the activation of the neurons in the input layer. Then the weighted sum of these neurons + the bias will determine the activation of the corresponding neuron in the next layer.
- The "hidden layers". The number of hidden layers and the number of neurons in them will be modified during the project to achieve a more accurate model, but for starters let's take 2 hidden layers with 16 neurons each. These hidden layers try to detect different patterns of the input. So for example the first hidden layer could try to recognize the edges of the the input, meaning we break the input into subproblems and the second layer could try to recognize the patterns that these edges make. The activations in one layer will determine the activations in the layer after it. So if the input layer activates the first hidden layers neurons that correspond to the relevant edges, then the first hidden layer would activate the second hidden layers neurons recognizing the relevant patterns of the edges and then the second hidden layer would activate the output layers neurons associated with the patterns. It doesn't actually do it in this way but it's an easy way of thinking of what it tries to do.
- The output layer. This is where we'll get the answer for what the neural network thinks that the given input is. This is done by taking the neuron with the biggest activation and the number corresponding to that neuron. So for a simple example let's take a situation where all of the other neurons have an activation of 0.01, but neuron number 10 has the activation of 0.98. In this situation the neural network is saying that it thinks that the input it saw is the digit 9. In our case the output layer consists of 10 neurons, since we have the possible outcomes of 0-9.

## Cost function
Each neuron is connected to all of the neurons in the previous layer. The weights and the weighted sum can be thought of as the "strength" between these connections. The bias indicates if the neuron is active or not. Now at first these weights and biases will be chosen at random and will need to be adjusted as we go. As you can imagine this randomness means that the network will not perform well at first.

This is where the cost function comes into play. It's a way for us to tell the computer that it's answer was bad. The smaller the cost the better the network is performing and vise versa.

So the cost function takes in as input all the weights and biases of the network and spits out a number representing how bad/good those weights and biases are. When we say that a network is learning it's just trying to minimize the cost function.

Example for a single training sample:
Let's imagine we give the network a picture of the number 3 as input. As explained above at first the output will be terrible since the weights and biases are chosen at random.

Now we take the output that the network gives us and the output we wanted the network to give us. So for example lets say that the fourth neuron in the output layer aka the neuron corresponding to the digit 3 has an activation of 0.24. The other outputs are let's say 0.5 for every neuron. Then we take the output we wanted it to give us, so the activation of 1 in the fourth neuron.

So the output of the network looks like this: (0.5, 0.5, 0.5, 0.24, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

But in reality we would want it to look like this: (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)

Now we take the sum of the squares of the differences between each component like this:

Cost of 3 = (0.5 - 0)^2 + (0.5 - 0)^2 + (0.5 - 0)^2 + (0.24 - 1)^2 + (0.5 - 0)^2 + (0.5 - 0)^2 +(0.5 - 0)^2 + (0.5 - 0)^2 +(0.5 - 0)^2 +(0.5 - 0)^2

This same process is then repeated for all the training samples and we take the average of the results to get the total cost of the network.

## Gradient descent
Gradient descent is the way used to minimize this cost function. You can think of it as a way of walking down a hill to the lowest local point. It tries to find the local minimum of the cost function by taking "steps" into each direction until it finds the minimum point. This is done by calculating the gradient of a function which gives us the the steepest direction to increase cost. Since the gradient gives us the direction to increase the cost the most we need to take the negative of it to lower it the most.

So to minimize the cost we need to compute the gradient direction, take a small step in to the other direction and repeat this over and over again until we find the minimum value.

## Backpropagation
Backpropagation is the algorithm used to compute this gradient in an efficient manner. To achieve the correct outcome we need to have the corresponding output layers neurons activation as high as possible and the remaining neurons in the output layer as low as possible. These activation can be changed by changing the activations of the connected neurons in the previous layer by altering the connections weights and biases. 

For example say our input was the digit 3. Now on our output layers neuron 4 will have some activation. To make this activation higher we need to make the activation of the neurons connected to neuron 4 with positive weight higher and the activation of the connected neurons with a negative weight smaller. This is how we can boost the activation of the correct output layers neuron. For all the other neurons in the output layer we want the activation to be as small as possible since, neuron 4 corresponds to the correct answer and the others dont. Taking into account the desires of each of the output layers neurons to make the previous layers neurons activation higher or lower we get an idea what should happen to each of the neurons in the previous layer.

Now we do this same step for each of the layers and this is called propagating backwards and where the name backpropagation comes from.

If we do this for all of the tens of thousands of training examples it would take alot of computing power and would be super slow, so instead we'll use something called stochastic gradient descent. This is where we randomly divide the data into "mini-batches" and compute each step for the mini-batch.

Time complexity for this depends heavily on the amount of layers in the network and the amount of neurons in these layers. The plan is to have 4 layers with 784 neurons in the input layer, 16 neurons in each of the two hidden layers and 10 neurons in the output layer. This would give us a big O notation of <b>O(nt*(ij+2j+jk))</b>, where <b>n</b> is the number of epochs (training rounds), <b>t</b> is the number of training examples and <b>i, j and k</b> represent the amount of neurons in each of the layers in the network.

## Course details
Im a computer science student at the University of Helsinki. I'll be completing my project in Python, but I have experience in other languages as well so peer reviewing projects in other languages is no issue.
