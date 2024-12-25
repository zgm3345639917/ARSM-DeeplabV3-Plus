import cv2
import numpy as np

# 读取标签图片
label_image = cv2.imread('Cityscapes/cityscapes_19classes_train/aachen_000000_000019.png', cv2.IMREAD_GRAYSCALE)

# 将标签图片转化为数组
label_array = np.array(label_image)

# 查看不同类别标签值
unique_labels = np.unique(label_array)
print("不同类别标签值:", unique_labels)