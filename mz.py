import os

# 标签图像所在的目录
label_images_dir = 'Cityscapes/cityscapes_19classes_tvt'

# 要替换的字符串和替换后的字符串
old_string = '_leftImg8bit'  # 要替换的原始字符串
new_string = ''  # 想要替换成的新字符串

# 遍历标签图像目录中的所有文件
for filename in os.listdir(label_images_dir):
    # 检查文件是否是图像文件（可以根据需要添加更多条件）
    if filename.endswith('.jpg'):
        # 构造文件的完整路径
        old_file_path = os.path.join(label_images_dir, filename)

        # 替换文件名中的字符串
        new_filename = filename.replace(old_string, new_string)

        # 构造新文件的完整路径
        new_file_path = os.path.join(label_images_dir, new_filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed {old_file_path} to {new_file_path}')