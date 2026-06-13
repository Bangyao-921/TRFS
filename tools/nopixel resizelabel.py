import os
import numpy as np
from PIL import Image

# 输入和输出路径
label_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/Z5"
save_label_path = r"/home/cc/sby/RGBX_Semantic_Segmentation-main/datasets/STSD/Z6"

# 确保输出目录存在
os.makedirs(save_label_path, exist_ok=True)

# 设置目标大小
target_size = (1536, 1024)  # 预设的目标大小


# 定义最近邻插值函数
def resize_no_new_pixel(src_img, out_h, out_w):
    dst_img = np.zeros((out_h, out_w), dtype=src_img.dtype)
    height = src_img.shape[0]
    width = src_img.shape[1]
    w_scale = float(width) / out_w
    h_scale = float(height) / out_h

    for j in range(out_h):
        for i in range(out_w):
            raw_w = int(i * w_scale)
            raw_h = int(j * h_scale)
            dst_img[j, i] = src_img[raw_h, raw_w]
    return dst_img


# 遍历目录中的每个 PNG 文件
for label_filename in os.listdir(label_path):
    if not label_filename.lower().endswith('.png'):
        continue

    # 打开标签图像
    label = Image.open(os.path.join(label_path, label_filename))

    # 将PIL图像转换为numpy数组
    src_img = np.array(label)

    # 使用最近邻插值调整标签的大小
    dst_img = resize_no_new_pixel(src_img, target_size[1], target_size[0])

    # 将numpy数组转换回PIL图像
    dst_img_pil = Image.fromarray(dst_img.astype(np.uint8))

    # 创建输出文件名
    basename = os.path.splitext(label_filename)[0]
    label_new_filename = f'{basename}.png'

    # 保存调整大小后的标签
    dst_img_pil.save(os.path.join(save_label_path, label_new_filename))

print("Label resizing completed without generating new values.")