# Test document

Did not test every single line since running tests on the backrpopagation would be the same as just running the network itself.

Unit test coverage can be found [here](https://luukasmakila.github.io/neural-network-to-recognize-digits/)

### What has been tested?

- Feedforward: Used dummy values for weights and biases, then compare the expected result we get staright from the sigmoid function to the result we get from feedworward.

- Sigmoid and Sigmoid_prime (sigmoids derivative): Check if the returned value is within the tolerance range specified by rtol and atol.

- Computing the cost derivative using the neural networks method: Computes the result and checks if the result is correct shape and if the values are what was expected with the given input.

### How to run the tests?

```console
pytest
```
