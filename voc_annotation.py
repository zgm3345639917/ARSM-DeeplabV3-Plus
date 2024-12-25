import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
# trainval_percent    = 1
# train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
Cityscapes_path      = 'Cityscapes'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in Segmentation.")                            # 正在“在Segmentation中生成txt文件”
    segfilepath1 = os.path.join(Cityscapes_path, 'Cityscapes_train')  # 将Cityscapes_path和'Cityscapes_train'连接起来，得到完整文件路径
    segfilepath2 = os.path.join(Cityscapes_path, 'Cityscapes_val')    # 将Cityscapes_path和'Cityscapes_val'连接起来，得到完整文件路径
    segfilepath3 = os.path.join(Cityscapes_path, 'Cityscapes_test')   # 将Cityscapes_path和'Cityscapes_test'连接起来，得到完整文件路径
    #segfilepath4 = os.path.join(Cityscapes_path, 'cityscapes_19classes_train')
    #segfilepath5 = os.path.join(Cityscapes_path, 'cityscapes_19classes_val')
    #segfilepath6 = os.path.join(Cityscapes_path, 'cityscapes_19classes_test')
    segfilepath7 = os.path.join(Cityscapes_path, 'cityscapes_19classes_tvt')
    saveBasePath = os.path.join(Cityscapes_path, 'Segmentation')

    temp_train = os.listdir(segfilepath1)   # 列出segfilepath路径下的所有文件和文件夹的名字，并将这些名字作为字符串列表赋值给temp_train变量。
    total_train = []                        # 初始化一个空列表total_train，用于存储满足特定条件的文件名
    for train in temp_train:                # 遍历temp_train列表中的每一个元素（即每一个文件名），每次循环时，当前的文件名会被赋值给变量train
        if train.endswith(".jpg"):          # 如果文件名以.png结尾，则执行下一行代码
            total_train.append(train)       # 将train添加到total_train列表中

    temp_val = os.listdir(segfilepath2)
    total_val = []
    for val in temp_val:
        if val.endswith(".jpg"):
            total_val.append(val)

    temp_test = os.listdir(segfilepath3)
    total_test = []
    for test in temp_test:
        if test.endswith(".jpg"):
            total_test.append(test)
    '''
    temp_train_label = os.listdir(segfilepath4)    # 列出segfilepath4路径下的所有文件和文件夹的名字，并将这些名字作为字符串列表赋值给temp_train_label变量。
    total_train_label = []                         # 初始化一个空列表total_train_label，用于存储满足特定条件的文件名
    for train_label in temp_train_label:           # 遍历temp_train_label列表中的每一个元素（即每一个文件名），每次循环时，当前的文件名会被赋值给变量train
        if train_label.endswith(".png"):           # 如果文件名以.png结尾，则执行下一行代码
            total_train_label.append(train_label)  # 将train_label添加到total_train_label列表中

    temp_val_label = os.listdir(segfilepath5)
    total_val_label = []
    for val_label in temp_val_label:
        if val_label.endswith(".png"):
            total_val_label.append(val_label)

    temp_test_label = os.listdir(segfilepath6)
    total_test_label = []
    for test_label in temp_test_label:
        if test_label.endswith(".png"):
            total_test_label.append(test_label)
    '''
    temp_tvt_label = os.listdir(segfilepath7)
    total_tvt_label = []
    for tvt_label in temp_tvt_label:
        if tvt_label.endswith(".png"):
            total_tvt_label.append(tvt_label)

    num1 = len(total_train)                 # 计算分割类图像的总数
    num2 = len(total_val)                   # 计算分割类图像的总数
    num3 = len(total_test)                  # 计算分割类图像的总数
    #num4 = len(total_train_label)
    #num5 = len(total_val_label)
    #num6 = len(total_test_label)
    num7 = len(total_tvt_label)
    list1 = range(num1)                     # 创建一个与图像数量相等的整数范围列表
    list2 = range(num2)                     # 创建一个与图像数量相等的整数范围列表
    list3 = range(num3)                     # 创建一个与图像数量相等的整数范围列表
    #list4 = range(num4)
    #list5 = range(num5)
    #list6 = range(num6)
    list7 = range(num7)
    '''
    tv = int(num*trainval_percent)
    tr = int(tv*train_percent)           # 根据trainval_percent（训练验证集占总数的比例）和train_percent（训练集占训练验证集的比例）计算出训练验证集和训练集的图像数量。
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)   # 从整数范围列表中随机抽取指定数量的元素，生成训练验证集和训练集。
    '''
    print("train size", num1)
    print("val size", num2)
    print("test size", num3)              # 打印训练集、验证集与测试集的大小
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')   # 使用open函数创建或覆盖三个文本文件，分别用于存储训练集、验证集和测试集的文件名。
    
    for i in list1:                      # 遍历之前创建的整数范围列表
        name = total_train[i][:-4]+'\n'  # 从total_seg列表中取出第i个文件名（即第i个分割类图像的文件名），并去掉其扩展名.png（通过切片操作[:-4]实现）。然后，在文件名后添加一个换行符\n，以便在写入文件时每个文件名占一行。
        ftrain.write(name)               # 将该文件名写入train.txt文件。
    for i in list2:                      # 遍历之前创建的整数范围列表
        name = total_val[i][:-4]+'\n'    # 从total_seg列表中取出第i个文件名（即第i个分割类图像的文件名），并去掉其扩展名.png（通过切片操作[:-4]实现）。然后，在文件名后添加一个换行符\n，以便在写入文件时每个文件名占一行。
        fval.write(name)                 # 将该文件名写入val.txt文件。
    for i in list3:                      # 遍历之前创建的整数范围列表
        name = total_test[i][:-4]+'\n'   # 从total_seg列表中取出第i个文件名（即第i个分割类图像的文件名），并去掉其扩展名.png（通过切片操作[:-4]实现）。然后，在文件名后添加一个换行符\n，以便在写入文件时每个文件名占一行。
        ftest.write(name)                # 将该文件名写入test.txt文件。
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in Segmentation done.")            # 在Segmentation文件夹中生成txt文件的工作已经完成

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")

    classes_nums = np.zeros([256], int)                 # 初始化了一个大小为256的整数数组classes_nums，并将其所有元素设置为0
    for i in tqdm(list7):
        name = total_tvt_label[i]
        png_file_name = os.path.join(segfilepath7, name)    # 通过索引i从total_seg列表中获取一个文件名（或文件路径的一部分），然后使用os.path.join将这个文件名与segfilepath（一个文件夹路径）拼接起来，形成完整的文件路径
        if not os.path.exists(png_file_name):              # 检查png_file_name所指向的文件是否存在
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png = np.array(Image.open(png_file_name), np.uint8)   # 使用Image.open函数打开PNG图片文件，并将其转换为NumPy数组。np.uint8指定了数组的数据类型为无符号8位整数，
        if len(np.shape(png)) > 2:                            # 检查图片数组的维度。如果维度大于2
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        # 使用np.reshape(png, [-1])将图片数组展平为一维数组。np.bincount函数则用于统计一维数组中每个值（在这里是像素值，代表类别）出现的次数。minlength=256确保返回的数组长度为256，即使某些类别在图片中并未出现，它们的计数也会被初始化为0。最后，通过+=操作，将统计结果累加到classes_nums数组中
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")