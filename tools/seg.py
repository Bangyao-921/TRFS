import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import math

# 输入和输出路径
image_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/STSD4"
label_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/Z4"
save_image_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/STSD5"
save_label_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/Z5"
# 确保输出目录存在
os.makedirs(save_image_path, exist_ok=True)
os.makedirs(save_label_path, exist_ok=True)

# 设置裁剪大小
cut_size = 512  # 预设的块大小

# 遍历目录中的每个 PNG 文件
for image_filename in os.listdir(image_path):
    if not image_filename.lower().endswith('.png'):
        continue

    # 获取对应的标签文件名
    label_filename = image_filename  # 假设标签文件名与图像文件名相同

    # 打开图像和标签文件
    image = Image.open(os.path.join(image_path, image_filename))
    label = Image.open(os.path.join(label_path, label_filename))

    # 获取图像的宽和高
    width, height = image.size

    # 计算横向和纵向的裁剪数量
    x_count = math.ceil((width - cut_size) / cut_size) + 1
    y_count = math.ceil((height - cut_size) / cut_size) + 1

    # 遍历所有裁剪后的小图
    for y in tqdm(range(y_count), desc=image_filename):
        for x in range(x_count):
            # 计算当前小图的左上角坐标
            xoff = x * cut_size
            yoff = y * cut_size

            # 检查边缘情况，进行回退
            if xoff + cut_size > width:
                xoff = width - cut_size
            if yoff + cut_size > height:
                yoff = height - cut_size

            # 计算当前小图的宽和高
            xsize = min(cut_size, width - xoff)
            ysize = min(cut_size, height - yoff)

            # 裁剪图像和标签
            image_crop = image.crop((xoff, yoff, xoff + xsize, yoff + ysize))
            label_crop = label.crop((xoff, yoff, xoff + xsize, yoff + ysize))

            # 创建输出文件名
            basename = os.path.splitext(image_filename)[0]
            image_new_filename = f'{basename}_{x}_{y}.png'
            label_new_filename = f'{basename}_{x}_{y}.png'

            # 保存裁剪后的图像和标签
            image_crop.save(os.path.join(save_image_path, image_new_filename))
            label_crop.save(os.path.join(save_label_path, label_new_filename))

print("Image and label slicing completed.")