import os
import shutil

# 假设 self.dataset_path 是您的数据集根目录
# new_folder_path 是您想要将所有图片放入的新文件夹的路径
# new_folder_name 是新文件夹的名称，它将位于 self.dataset_path 下
dataset_path = "Cityscapes"
new_folder_name = "cityscapes_tv"
new_folder_path = os.path.join(dataset_path, new_folder_name)

# 确保新文件夹存在
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# 定义要从中复制图片的文件夹名称
source_folders = ["Cityscapes_train", "Cityscapes_val"]

# 遍历每个源文件夹，并复制其中的图片到新文件夹
for source_folder in source_folders:
    source_path = os.path.join(dataset_path, source_folder)
    if os.path.exists(source_path):
        for filename in os.listdir(source_path):
            if filename.lower().endswith('.jpg'):
                # 构建源文件和目标文件的完整路径
                source_file = os.path.join(source_path, filename)
                target_file = os.path.join(new_folder_path, filename)
                # 复制文件
                shutil.copy2(source_file, target_file)  # copy2 保留元数据
                print(f"Copied {source_file} to {target_file}")
    else:
        print(f"The folder {source_path} does not exist.")

print(f"All images have been copied to {new_folder_path}")