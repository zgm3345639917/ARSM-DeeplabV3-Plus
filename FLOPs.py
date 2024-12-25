import thop
import torch
from nets.deeplabv3_plus import DeepLab  # 以resnet18为例，你可以替换成你的模型

# 实例化模型
model = DeepLab(num_classes=19, backbone="mobilenet", pretrained=True, downsample_factor=16)   # xception mobilenet
model.eval()  # 确保模型在评估模式下

# 假设你有一个输入张量，其大小与模型期望的输入大小相匹配
input = torch.randn(8, 3, 512, 512)

# 使用thop来计算FLOPs和参数数量
flops, params = thop.profile(model, inputs=(input,), verbose=False)

print(f'Total FLOPs: {flops / 1e9} G')  # 将FLOPs转换为G（十亿次）
print(f'Total parameters: {params/1e6}M')