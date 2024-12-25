import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction):
        super(CoordAtt, self).__init__()
        # 定义两个自适应平均池化层，分别沿高度和宽度方向进行池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)  # 计算中间层的通道数，确保它不小于8

        # 定义一个卷积层，将输入特征图的通道数降维到mip
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)   # 定义一个批量归一化层
        self.act = h_swish()   # 定义激活函数，这里使用了一个名为h_swish的自定义激活函数，需要确保此函数已经定义

        # 定义两个卷积层，将mip通道的特征图转换为oup通道
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x                       # 保留原始输入特征图，用于后续计算输出

        n, c, h, w = x.size()              # 获取输入特征图的尺寸
        x_h = self.pool_h(x)               # 对输入特征图进行高度池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 对输入特征图进行宽度池化，并转置高度和宽度维度

        y = torch.cat([x_h, x_w], dim=2)   # 将高度和宽度池化后的特征图在通道维度上连接
        y = self.conv1(y)                  # 通过卷积层降维
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)  # 将特征图分割为高度和宽度的部分
        x_w = x_w.permute(0, 1, 3, 2)             # 将宽度的特征图转置回原来的维度顺序

        a_h = self.conv_h(x_h).sigmoid()          # 通过卷积层并应用sigmoid函数得到高度和宽度的注意力权重
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h                # 将原始输入特征图与注意力权重相乘，得到输出特征图
        return out
