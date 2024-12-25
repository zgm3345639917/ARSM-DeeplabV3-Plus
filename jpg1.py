import os
from PIL import Image

# 设置Cityscapes数据集原图的目录
original_images_dir1 = 'cityscapes(jpg)/Cityscapes_train'
original_images_dir2 = 'cityscapes(jpg)/Cityscapes_val'
original_images_dir3 = 'cityscapes(jpg)/Cityscapes_test'
# 设置转换后JPEG图像的保存目录
#output_dir1 = 'Cityscapes(jpg)/Cityscapes_train1'
#output_dir2 = 'Cityscapes(jpg)/Cityscapes_val1'
#output_dir3 = 'Cityscapes(jpg)/Cityscapes_test1'
output_dir = 'Cityscapes/Cityscapes_tvt'
'''
# 确保输出目录存在
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
#if not os.path.exists(output_dir3):
   # os.makedirs(output_dir3)
'''
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历原图目录中的所有文件
for filename in os.listdir(original_images_dir1):
    # 检查文件是否为图像文件（这里假设为.png，你可以根据需要修改）
    if filename.endswith('.png'):
        # 构建原图和输出文件的完整路径
        input_path1 = os.path.join(original_images_dir1, filename)
        output_path1 = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
# 使用Pillow打开图像并保存为JPEG格式
        with Image.open(input_path1) as img:
            rgb_img = img.convert('RGB')  # 如果原图不是RGB格式，这里转换为RGB
            rgb_img.save(output_path1, 'JPEG', quality=95)  # quality参数用于控制JPEG压缩质量，范围是0-100
        print(f"Converted {input_path1} to {output_path1}")

for filename in os.listdir(original_images_dir2):
    if filename.endswith('.png'):
        input_path2 = os.path.join(original_images_dir2, filename)
        output_path2 = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
        with Image.open(input_path2) as img:
            rgb_img = img.convert('RGB')  # 如果原图不是RGB格式，这里转换为RGB
            rgb_img.save(output_path2, 'JPEG', quality=95)  # quality参数用于控制JPEG压缩质量，范围是0-100
        print(f"Converted {input_path2} to {output_path2}")

for filename in os.listdir(original_images_dir3):
    if filename.endswith('.png'):
        input_path3 = os.path.join(original_images_dir3, filename)
        output_path3 = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
        with Image.open(input_path3) as img:
            rgb_img = img.convert('RGB')  # 如果原图不是RGB格式，这里转换为RGB
            rgb_img.save(output_path3, 'JPEG', quality=95)  # quality参数用于控制JPEG压缩质量，范围是0-100
        print(f"Converted {input_path3} to {output_path3}")
