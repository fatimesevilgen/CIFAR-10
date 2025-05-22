import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 4)
        self.pool  = nn.MaxPool2d(2, 2)

        self.fc1   = nn.Linear(12 * 6 * 6, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 12 * 6 * 6)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
