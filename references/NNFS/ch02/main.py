import numpy as np

# single neuron
inputs = [1.0, 2.0, 3., 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# output of a single neuron
outputs = np.dot(weights, inputs) + bias

# calculate the output of a layer of 3 neurons
inputs = [1.0, 2.0, 3., 2.5]  # (4,)
weights = [[0.2, 0.8, -0.5, 1.0],  # (3, 4)
           [0.5, -0.8, -0.4, -1.0],
           [-0.26, 0.9, -0.55, 1.7]]
biases = [2., 3., 0.5]

# a dot product method trats the matrix as a list of vectors
# and performs a dot product of each of those vectors with the other vector
# layer_outputs = np.dot(weights, inputs) + biases

# batch of data
# a matrix with one row
print(np.array([[2, 3, 4]]).shape)
# a matrix with one column (a vector)
print(np.array([2, 3, 4]).shape)

# we can turn use expand_dims numpy methods to create a matrix from a vector
a = [1, 2, 4]
print(np.array(a).shape)
# expand_dims add one new dimension at the index of the axis param
print(np.expand_dims(np.concatenate([np.array([a]), np.array([a])], axis=0), axis=0).shape)
print(np.concatenate([np.array([a]), np.array([a])], axis=0).shape)

# transposition
a = np.array([[1, 4, 5]])
b = np.array([4, 5, 65]).T
print(np.dot(a, b))

# our first layer with batch processing on 3 neurons layer
inputs2 = [[1., 2., 3.],  # (3, 3)
          [4., 5., 1.4],
          [3., 4., 1.1]]

weights2 = [[0.2, 0.8, 1.0],  # (4, 3)
           [0.5, -0.8, -1.0],
           [-0.26, 0.9, 1.7],
           [1., 1., 2.]]

biases2 = [2., 3., 0.5, 3]

layer_outputs = np.dot(inputs2, np.array(weights2).T) + biases2  # we see similar stuff on tf scratch
print(layer_outputs.shape)
print(layer_outputs)
