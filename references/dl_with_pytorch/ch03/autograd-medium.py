import torch

a = torch.tensor(10.0, requires_grad=True)
b = torch.tensor(20.0, requires_grad=True)

f = a * b

f.backward()

print(a.grad)
print(b.grad)

# now if we add a vector to a, we'll get an error
a = torch.tensor([10.0, 20.0], requires_grad=True)
b = torch.tensor([20.0, 20.0], requires_grad=True)

f = a * b

f.backward(gradient=torch.tensor([1.0, 1.0]))

print(a.grad)
print(b.grad)


# another example
a = torch.tensor([10.0, 20.0], requires_grad=True)
b = torch.tensor([20.0, 20.0], requires_grad=True)

f = a * b
g = 2 * f

g.backward(gradient=torch.tensor([1.0, 1.0]))

print(a.grad)
print(b.grad)
print(f.grad)
