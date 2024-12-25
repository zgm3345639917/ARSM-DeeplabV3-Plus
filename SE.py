import torch.nn as nn


class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化将输入特征图的宽和高都压缩为1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 将输入通道数从channel减少到channel // reduction
            nn.ReLU(inplace=True),  # inplace=True时，ReLU激活函数会直接修改输入张量，而不是返回一个新的张量。这可以节省内存，因为不需要为输出分配新的存储空间
            nn.Linear(channel // reduction, channel, bias=False),  # 将通道数恢复为channel
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()            # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 对输入x进行自适应平均池化，并将结果展平为二维张量
        y = self.fc(y).view(b, c, 1, 1)  # 将池化后的结果通过全连接网络，并重新塑形为(batch_size, channel, 1, 1)
        return x * y.expand_as(x)        # 将全连接网络的输出y扩展为与输入x相同的形状，并与x进行逐元素相乘。这是SE块的关键操作，允许模型重新调整不同通道的重要性。



