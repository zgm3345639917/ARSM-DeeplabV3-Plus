import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from ACBA import ACAConv2d1, ACAConv2d2, ACAConv2d3, ACAConv2d4
from SA import ShuffleAttention
from CoordAttention import CoordAtt


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]       # 索引从0开始

        if downsample_factor == 8:
            # 遍历从self.down_idx[-2]到self.down_idx[-1]（不包括self.down_idx[-1]）的索引，并对这些索引对应的self.features中的层应用_nostride_dilate方法，其中膨胀系数dilate为2。
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
                # 遍历从self.down_idx[-1]到self.total_idx（不包括self.total_idx）的索引，并对这些索引对应的self.features中的层应用_nostride_dilate方法，其中膨胀系数dilate为4。
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            # 遍历从self.down_idx[-1]到self.total_idx（不包括self.total_idx）的索引，并对这些索引对应的self.features中的层应用_nostride_dilate方法，其中膨胀系数dilate为2。
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:    # 检查classname是否包含字符串'Conv'。如果包含，find方法会返回该子字符串的起始索引；如果不包含，则返回-1。
            if m.stride == (2, 2):
                m.stride = (1, 1)           # 检查卷积层的步长是否为(2, 2)。如果步长为(2, 2)，则将其修改为(1, 1)。
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)   # 如果卷积核的大小为(3, 3)，则设置膨胀系数和填充为dilate的一半（取整）
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
       # low_level_features = self.features[:2](x)  # 256x256x16
        mid_level_features1 = self.features[:4](x)  # 128x128x24
        mid_level_features2 = self.features[:7](x)  # 64x64x32
        high_level_features = self.features[4:](mid_level_features1)  # 32x32x320
        return mid_level_features1, mid_level_features2, high_level_features


'''
    def forward(self, x):
        low_level_features = x
        high_level_features = None
        for layer in self.features[:4]:
            low_level_features = layer(low_level_features)      # 128x128x24 使用模型的前4个特征提取层（或模块）处理输入 x，并获取低级特征
        for layer in self.features[4:]:
            high_level_features = layer(low_level_features)     # 32x32x320 使用模型的其余特征提取层处理低级特征 low_level_features，并获取高级特征
        return low_level_features, high_level_features
    '''
# -----------------------------------------#
#   ARS_ASPP特征提取模块
#   利用非对称空洞卷积ACBA进行特征提取
# -----------------------------------------#


class ARS_ASPP(nn.Module):
    def __init__(self, dim_in=320, dim_out=256, rate=1, bn_mom=0.1):
        super(ARS_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            ACAConv2d1(320, 256),
            nn.BatchNorm2d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            ACAConv2d2(320, 256),
            nn.BatchNorm2d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            ACAConv2d3(320, 256),
            nn.BatchNorm2d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            ACAConv2d4(320, 256),
            nn.BatchNorm2d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch6_conv = nn.Conv2d(320, 256, 1, 1, 0, bias=True)
        self.branch6_bn = nn.BatchNorm2d(256, momentum=bn_mom)
        self.branch6_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(256 * 6, 256, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.sa = ShuffleAttention(channel=1536)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        conv3x3_4 = self.branch5(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch6_conv(global_feature)
        global_feature = self.branch6_bn(global_feature)
        global_feature = self.branch6_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature], dim=1)
        sa_out = self.sa(feature_cat)
        result = self.conv_cat(sa_out)
        return result    # 32x32x256 深层


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [32,32,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 16
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        self.aspp = ARS_ASPP(dim_in=in_channels, dim_out=256, rate=1, bn_mom=0.1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.ca1 = CoordAtt(352, 352, 4)
        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(352, 256, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.ca2 = CoordAtt(304, 304, 4)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        mid_level_features1, mid_level_features2, high_level_features = self.backbone(x)
        x = high_level_features
        x = self.aspp(x)  # 32x32x256

        # 2倍上采样 中间层2调整通道为96 进入ca注意力 拼接 1x1conv调整通道 64x64x256
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        mid_level_features2 = self.conv1(mid_level_features2)
        x1 = torch.cat([mid_level_features2, x], dim=1)
        ca1_out = self.ca1(x1)
        x = self.conv_cat1(ca1_out)  # 64x64x256

        # 2倍上采样 浅层调整通道为48 拼接进入ca注意力 1x1conv调整通道 128x128x256
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        mid_level_features1 = self.conv2(mid_level_features1)
        x2 = torch.cat([mid_level_features1, x], dim=1)
        ca2_out = self.ca2(x2)
        x = self.cat_conv(ca2_out)
        '''
        # 2倍上采样 浅层调整通道为48 拼接进入sa注意力 1x1conv调整通道 128x128x256
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        low_level_features = self.conv3(low_level_features)
        low_level_features = self.ca3(low_level_features)
        x = self.cat_conv(torch.cat([low_level_features, x], dim=1))
        '''
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x