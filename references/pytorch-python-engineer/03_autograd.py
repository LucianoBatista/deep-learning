import torch

x = torch.randn(3, requires_grad=True)

y = x + 2
z = y * x
# f = z.mean()
# here z is not a scalar, and because of that we need to pass a vector with the same dimensions
# of x.

v = torch.tensor([0.1, 1.0, 0.01], dtype=torch.float32)
z.backward(v)  # dz/dx
print(f"x {x}")
print(f"x {x.grad}")
print(f"z {z}")

# ----ways to eliminate the need of grad on a vector-----------------------
x2 = torch.randn(3, requires_grad=True)
# print(x2.requires_grad_(False))
# print(x2.detach())
with torch.no_grad():
    y2 = x2 + 2
    # print(y2)

# ----dummy example---------------------------------------------------------
weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)

    # very important to clean your gradients
    # if you do not do that you'll get into problems
    weights.grad.zero_()
