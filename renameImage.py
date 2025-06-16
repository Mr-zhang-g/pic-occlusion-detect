import os
from datetime import datetime

# 定义要重命名的文件夹路径
folder_path = r'D:\Data\人工智能\人脸数据集\with_mask'

# 获取该文件夹下的所有文件
files = os.listdir(folder_path)
# 初始化计数器
counter = 1

# 遍历文件
for file in files:
    # 只处理以.jpg或.png结尾的文件
    if file.endswith('.jpg') or file.endswith('.png'):
        # 定义新的文件名
        new_name = f"faceWithMask_{counter}.jpg"
        # 获取完整路径
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} to {new_file_path}")

        # 更新计数器
        counter += 1

