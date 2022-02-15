# dense layers are also called of fully-connected
import numpy as np
from nnfs.datasets import spiral_data


class LayerDense:

    # layer init
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # random numbers from a gaussian distribution
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        # calculate output values from inputs, weights and biases
        output = np.dot(inputs, self.weights) + self.biases
        return output


X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(len(X[0]), 3)
output_dense1 = dense1.forward(X)
