import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):
    """
    Canonical TrackNet implementation
    Input : [B, 9, H, W]  (3 RGB frames)
    Output: [B, 1, H, W]  (Gaussian heatmap)
    """

    def __init__(self, input_channels=9):
        super(TrackNet, self).__init__()

        # ---------------- Encoder ----------------
        self.conv1 = self._block(input_channels, 64, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self._block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self._block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self._block(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2)

        # ---------------- Bottleneck ----------------
        self.conv5 = self._block(512, 512, 3)

        # ---------------- Decoder ----------------
        self.up4 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec4 = self._block(256, 256, 2)

        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec3 = self._block(128, 128, 2)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = self._block(64, 64, 2)

        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = self._block(32, 32, 1)

        # ---------------- Output ----------------
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _block(self, in_ch, out_ch, layers):
        modules = []
        for i in range(layers):
            modules.append(
                nn.Conv2d(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1
                )
            )
            modules.append(nn.BatchNorm2d(out_ch))
            modules.append(nn.ReLU(inplace=True))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)

        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))

        return self.out(x)