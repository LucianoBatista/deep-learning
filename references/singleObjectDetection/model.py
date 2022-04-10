import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self, params) -> None:
        super(Net, self).__init__()
        c_in, h_in, w_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_outputs = params["num_outputs"]

        self.conv1 = nn.Conv2d(c_in, init_f, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            init_f + c_in, 2 * init_f, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            3 * init_f + c_in, 4 * init_f, kernel_size=3, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            7 * init_f + c_in, 8 * init_f, kernel_size=3, stride=2, padding=1
        )
        self.conv5 = nn.Conv2d(
            15 * init_f + c_in, 16 * init_f, kernel_size=3, stride=2, padding=1
        )
        self.fc1 = nn.Linear(16 * init_f, num_outputs)

    def forward(self, x):
        identity = f.avg_pool2d(x, 4, 4)
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = f.avg_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = f.avg_pool2d(x, 2, 2)
        x = f.relu(self.conv3(x))
        x = f.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = f.avg_pool2d(x, 2, 2)
        x = f.relu(self.conv4(x))
        x = f.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        x = f.relu(self.conv5(x))
        x = f.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)

        return x
