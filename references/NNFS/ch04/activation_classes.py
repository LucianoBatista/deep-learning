from nnfs.datasets import spiral_data
import numpy as np

from ch03.dense_layer import LayerDense


class Activation:

    def __init__(self, inputs):
        self.inputs = inputs

    def relu(self):
        output = np.maximum(0, self.inputs)
        return output

    def softmax(self):
        # trying to minimize extreme values
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True))
        print(exp_values)
        # normalize
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
dense1_out = dense1.forward(X)
activation = Activation(dense1_out)
relu_out = activation.relu()
softmax_out = activation.softmax()

print(softmax_out)
