# from scratch
import numpy as np

# f = w.x
# f = 2.x

X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


# gradient
# dj/dw = 1/N 2.x (w.x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(y, y_pred)
    dw = gradient(X, y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
