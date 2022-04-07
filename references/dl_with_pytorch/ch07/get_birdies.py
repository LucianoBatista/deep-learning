import torch
import torch.nn as nn
import torch.optim as optim

# model
model = nn.Sequential(
    nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.LogSoftmax(dim=1)
)

# learning rate
lr = 1e-2

# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# loss
loss_fn = nn.NLLLoss()

# epochs
n_epochs = 100

# forward pass
for epoch in range(n_epochs):
    for img, label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out, torch.tensor([label]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: {}, Loss: {}".format(epoch, loss))
