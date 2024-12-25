import torch.nn as nn


class RDConv(nn.Module):
    def __init__(self, in_channels=304, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(RDConv, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 残差连接
        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        residual = self.residual_connection(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.BN(x)
        x = self.relu(x)
        x = x + residual
        return x

