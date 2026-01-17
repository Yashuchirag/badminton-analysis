import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackNet(nn.Module):  
    def __init__(self):
        super(TrackNet, self).__init__()

        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Input:  (B, 9, H, W)
        Output: (B, 1, H, W)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.sigmoid(self.conv5(x))
        return x
