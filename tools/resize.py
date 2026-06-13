import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import math

# 输入和输出路径
image_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/STSD1"
label_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/H"
save_image_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/STSD5"
save_label_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/H4"

# 确保输出目录存在
os.makedirs(save_image_path, exist_ok=True)
os.makedirs(save_label_path, exist_ok=True)

# 设置目标大小
target_size = (1536, 1024)  # 预设的目标大小

# 遍历目录中的每个 PNG 文件
for image_filename in os.listdir(image_path):
    if not image_filename.lower().endswith('.png'):
        continue

    # 获取对应的标签文件名
    label_filename = image_filename  # 假设标签文件名与图像文件名相同

    # 打开图像和标签文件
    image = Image.open(os.path.join(image_path, image_filename))
    label = Image.open(os.path.join(label_path, label_filename))

    # 调整图像和标签的大小
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    label_resized = label.resize(target_size, Image.Resampling.LANCZOS)

    # 创建输出文件名
    basename = os.path.splitext(image_filename)[0]
    image_new_filename = f'{basename}.png'
    label_new_filename = f'{basename}.png'

    # 保存调整大小后的图像和标签
    image_resized.save(os.path.join(save_image_path, image_new_filename))
    label_resized.save(os.path.join(save_label_path, label_new_filename))

print("Image and label resizing completed.")