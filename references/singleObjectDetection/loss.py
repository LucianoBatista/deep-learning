from torch import optim
import torch
import torch.nn as nn
from model import Net

# loss
loss_func = nn.SmoothL1Loss(reduction="sum")

n, c = 8, 2
y = 0.5 * torch.ones(n, c, requires_grad=True)
print(y.shape)

target = torch.zeros(n, c, requires_grad=False)
print(target.shape)

loss = loss_func(y, target)
print(loss.item())

y = 2 * torch.ones(n, c, requires_grad=True)
target = torch.zeros(n, c, requires_grad=False)
loss = loss_func(y, target)
print(loss.item())

# optimizer
params_model = {
    "input_shape": (3, 256, 256),
    "initial_filters": 16,
    "num_outputs": 2,
}

model = Net(params_model)
opt = optim.Adam(model.parameters(), lr=3e-4)


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


current_lr = get_lr(opt)
print("Current lr: {}".format(current_lr))
