import torch

print(torch.cuda.is_available())

device = torch.device("cuda")
x = torch.rand(3, 3).to(device)

print(print(x))
