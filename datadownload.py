import os
import shutil
import random
from glob import glob

# 定义数据集路径
parent_dir = "Dataset"  # 修改为你的实际路径
rgb_folder = os.path.join(parent_dir, "RGB")  # RGB 目录
train_pos_dir = os.path.join(parent_dir, "positives")      # 训练集-正样本
val_pos_dir = os.path.join(parent_dir, "positives_t")      # 验证集-正样本
train_neg_dir = os.path.join(parent_dir, "negatives")      # 训练集-负样本
val_neg_dir = os.path.join(parent_dir, "negatives_t")      # 验证集-负样本

# 创建数据集目录
for folder in [train_pos_dir, val_pos_dir, train_neg_dir, val_neg_dir]:
    os.makedirs(folder, exist_ok=True)

# 获取所有图片
positive_images = sorted(glob(os.path.join(rgb_folder, "positives*.png")))  # 正样本
negative_images = sorted(glob(os.path.join(rgb_folder, "negatives*.png")))  # 负样本

# 设置训练/验证集比例（80% 训练, 20% 验证）
train_ratio = 0.8
train_pos_count = int(len(positive_images) * train_ratio)
train_neg_count = int(len(negative_images) * train_ratio)

# 打乱数据
random.shuffle(positive_images)
random.shuffle(negative_images)

# 数据划分
train_pos, val_pos = positive_images[:train_pos_count], positive_images[train_pos_count:]
train_neg, val_neg = negative_images[:train_neg_count], negative_images[train_neg_count:]

# 复制文件函数
def copy_files(file_list, target_folder):
    for file_path in file_list:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(target_folder, filename)
        shutil.copy2(file_path, dest_path)  # 复制文件
        print(f"Copied: {file_path} -> {dest_path}")

# 复制正样本和负样本到训练/验证集
copy_files(train_pos, train_pos_dir)
copy_files(val_pos, val_pos_dir)
copy_files(train_neg, train_neg_dir)
copy_files(val_neg, val_neg_dir)

print("数据集划分完成！")
