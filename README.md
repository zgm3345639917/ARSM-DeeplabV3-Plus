## ARSM-DeepLabV3-Plus
---

### 目录
1. [相关仓库 Related code](#相关仓库)
2. [性能情况 Performance](#性能情况)
3. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [预测步骤 How2predict](#预测步骤)
7. [评估步骤 miou](#评估步骤)
8. [参考资料 Reference](#Reference)

1## 相关仓库
| 模型 | 路径 |
| :----- | :----- |
ARSM-DeeplabV3-Plus | https://github.com/zgm3345639917/ARSM-DeeplabV3-Plus

2### 性能情况
| 训练数据集 | 预训练权值文件名称 | 测试数据集 | 输入图片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| cityscapes_train | [deeplab_mobilenetv2.pth](https://www.123865.com/s/BC5eTd-A87JH提取码:1234) | cityscapes_val | 512x512| 66.98 | 
| cityscapes_train | [deeplab_xception.pth](https://www.123865.com/s/BC5eTd-A87JH提取码:1234) | cityscapes_val | 512x512| 68.25 | 

3### 所需环境
torch==2.0.1    

4### 文件下载
训练所需的deeplab_mobilenetv2.pth和deeplab_xception.pth可在123网盘中下载。     
链接: https://www.123865.com/s/BC5eTd-A87JH提取码:1234
下载好预训练权重后，可以新建一个“model_data”的文件夹，将2个权重放入此文件夹中。 

Cityscapes数据集的123网盘如下：  
链接: https://www.123865.com/s/BC5eTd-h87JH提取码:3333   

5### 训练步骤
#### a、训练Cityscapes数据集
1、将我提供的Cityscapes数据集直接以“Cityscapes”命名文件夹即可（无需运行voc_annotation.py）。  
2、在train.py中设置对应参数，默认参数已经对应Cityscapes数据集所需要的参数了，所以只要修改backbone和model_path即可。 在train.py程序中将“model_data/deeplab_mobilenetv2.pth”写入model_path = ""。  
3、运行train.py进行训练。 
4、训练时会自动生成logs文件夹，训练的权重都存放在其中。

6### 预测步骤    
1、按照训练步骤训练。    
2、在deeplab.py文件里面，在如下部分修改model_path、num_classes、backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，num_classes代表要预测的类的数量，backbone是所使用的主干特征提取网络**。    
```python
_defaults = {
    #----------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #----------------------------------------#
    "model_path"        : 'logs/best_epoch_weights.pth',
    #----------------------------------------#
    #   所需要区分的类的个数（没有算入background，所以不需classes+1）
    #----------------------------------------#
    "num_classes"       : 19,
    #----------------------------------------#
    #   所使用的的主干网络
    #----------------------------------------#
    "backbone"          : "mobilenet",
    #----------------------------------------#
    #   输入图片的大小
    #----------------------------------------#
    "input_shape"       : [512, 512],
    #----------------------------------------#
    #   下采样的倍数，一般可选的为8和16
    #   与训练时设置的一样即可
    #----------------------------------------#
    "downsample_factor" : 16,
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"             : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3、运行predict.py，输入    
```python
img/1.jpg
（img文件夹中的图片都可以用来预测，自己也可以添加图片）
```
可完成预测。    
4、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。   

7### 评估步骤
1、设置get_miou.py里面的num_classes为预测的类的数量19。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  
1、2两步其实已经设置好了，所以只需要进行第3步，运行get_miou.py即可，会生成一个miou_out文件夹，里面有miou,mpa等性能参数

8### Reference
https://github.com/bubbliiiing/deeplabv3-plus-pytorch
