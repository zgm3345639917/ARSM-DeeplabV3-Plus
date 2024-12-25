import torch.nn as nn


class ACAConv2d1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn_mom=0.1):
        super(ACAConv2d1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=6, dilation=6, bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        x1 = self.conv(x)
        ac1 = self.ac1(x)
        ac2 = self.ac2(x)
        residual = self.residual(x)
        x = x1 + ac1 + ac2 + residual
        return x


class ACAConv2d2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn_mom=0.1):
        super(ACAConv2d2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=12, dilation=12, bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        x1 = self.conv(x)
        ac1 = self.ac1(x)
        ac2 = self.ac2(x)
        residual = self.residual(x)
        x = x1 + ac1 + ac2 + residual
        return x


class ACAConv2d3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn_mom=0.1):
        super(ACAConv2d3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=18, dilation=18, bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        x1 = self.conv(x)
        ac1 = self.ac1(x)
        ac2 = self.ac2(x)
        residual = self.residual(x)
        x = x1 + ac1 + ac2 + residual
        return x


class ACAConv2d4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn_mom=0.1):
        super(ACAConv2d4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=24, dilation=24, bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.ac2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), dilation=1,
                      bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        x1 = self.conv(x)
        ac1 = self.ac1(x)
        ac2 = self.ac2(x)
        residual = self.residual(x)
        x = x1 + ac1 + ac2 + residual
        return x

    '''
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=24, dilation=24,
                                      bias=True)
                self.ac1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), 
                                     dilation=1,bias=True)
                self.ac2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), 
                                     dilation=1,bias=True)
    '''