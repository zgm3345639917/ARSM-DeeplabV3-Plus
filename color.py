import numpy as np
from PIL import Image

# 假设你有一个灰度图像，其中每个像素的灰度值代表一个类别索引
gray_image = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])

# 定义每个类别对应的颜色（RGB值）
color_map = {
    0: (255, 0, 0),  # 红色
    1: (0, 255, 0),  # 绿色
    2: (0, 0, 255)  # 蓝色
}

# 初始化一个与灰度图像大小相同的彩色图像数组
colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

# 应用颜色映射
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        category_index = gray_image[i, j]
        colored_image[i, j] = color_map[category_index]

    # 将NumPy数组转换为PIL图像并显示
colored_image_pil = Image.fromarray(colored_image)
colored_image_pil.show()