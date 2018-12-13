import torch.nn as nn
import torch.nn.functional as F


class SleepCNN_1D(nn.Module):
    def __init__(self):
        super(SleepCNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=25, stride=10)
        self.pool = nn.MaxPool1d(kernel_size=20, stride=2)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=25, stride=10)
        self.fc1 = nn.Linear(in_features=16 * 27, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 27)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SleepCNN_1D_Tsinalis(nn.Module):
    """Replication of the CNN architecture used by Tsinalis,
    et al. (2006)
    """

    def __init__(self):
        super(SleepCNN_1D_Tsinalis, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=200, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=20, stride=10)
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=400, kernel_size=(20, 30), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 2))
        self.fc1 = nn.Linear(in_features=400 * 721, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=5)

    def forward(self, x):
        # One dimensional convolution/pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Stack, two-dimensional convolution/pooling
        x = x.unsqueeze(1)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Reshape, fully connected layers
        x = x.view(-1, 400 * 721)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
