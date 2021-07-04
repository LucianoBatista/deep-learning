# activation functions
# step function
# linear activation function: straight line
# sigmoid function: -inf = 0, +inf = 1
# we want that activation functions show us some information about how close we are from the true.
# relu: negative = 0, positive = positive
# relu is unit unbounded, not normalized and exclusive
# softmax: can take in non-normalized, or uncalibrated, inputs and produce a normalized distribution of
# probabilities for our classes.

import numpy as np


inputs = [0, 2, -1, 3.3, 2.6, -100, 25]
output = []

matrix = [[1, 2, 3],
          [2, 3, 4],
          [5, 6, 7]]

# 0 = column, 1 = rows. But matrices are (row, column)
for i in inputs:
    output.append(np.maximum(0, i))

# softmax
print(np.exp(matrix))
print(np.max(matrix, axis=1, keepdims=True))
print(np.exp(1))