import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_dolp_npy_to_image(src_folder, dst_folder, mode='gray'):
    """
    将 .npy 格式的 DoLP 数据转换为图像。

    Args:
        src_folder (str): 存放 .npy 文件的文件夹路径
        dst_folder (str): 保存图像的文件夹路径
        mode (str): 'gray' 输出灰度图, 'color' 输出热力图 (伪彩色)
    """

    # 如果目标文件夹不存在，则创建
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"已创建目标文件夹: {dst_folder}")

    files = [f for f in os.listdir(src_folder) if f.endswith('.npy')]
    total_files = len(files)

    print(f"开始处理，共发现 {total_files} 个 npy 文件...")

    for i, filename in enumerate(files):
        # 1. 加载数据
        npy_path = os.path.join(src_folder, filename)
        try:
            # 加载 .npy 文件
            dolp_data = np.load(npy_path)

            # 检查数据维度，DoLP 通常是 2D 矩阵 (H, W)
            # 如果是 (H, W, 1) 则去掉最后一维
            if dolp_data.ndim == 3 and dolp_data.shape[2] == 1:
                dolp_data = dolp_data.squeeze()

        except Exception as e:
            print(f"加载文件 {filename} 失败: {e}")
            continue

        # 2. 数据清洗与归一化
        # 处理可能存在的 NaN (无效值) 或 Inf (无穷大)
        dolp_data = np.nan_to_num(dolp_data, nan=0.0, posinf=1.0, neginf=0.0)

        # 理论上 DoLP 范围是 [0, 1]。
        # 为了安全起见，我们进行截断操作，防止噪点导致数值溢出
        dolp_data = np.clip(dolp_data, 0, 1)

        # 3. 图像转换
        output_filename = os.path.splitext(filename)[0] + '.png'
        save_path = os.path.join(dst_folder, output_filename)

        if mode == 'gray':
            # --- 模式 A: 灰度图 (0-255 uint8) ---
            # 适合作为神经网络的输入
            img_uint8 = (dolp_data * 255).astype(np.uint8)
            cv2.imwrite(save_path, img_uint8)

        elif mode == 'color':
            # --- 模式 B: 伪彩色热力图 (Jet/Inferno) ---
            # 适合人类视觉检查，高偏振度显示为暖色
            # 使用 Matplotlib 的 colormap
            plt.imsave(save_path, dolp_data, cmap='inferno', vmin=0, vmax=1)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{total_files} 个文件")

    print(f"处理完成！所有图像已保存至: {dst_folder}")


# ================= 使用示例 =================

# 请修改为您实际的路径
source_directory = '/home/cc/sby/mcubes/polL_dolp'  # 您的 .npy 文件所在目录
output_directory_gray = '/home/cc/sby/mcubes/dolp'  # 输出灰度图目录
output_directory_color = '/home/cc/sby/mcubes'  # 输出彩色图目录

# 1. 生成灰度图 (推荐用于训练)
convert_dolp_npy_to_image(source_directory, output_directory_gray, mode='gray')

# 2. 生成热力图 (推荐用于观察)
# convert_dolp_npy_to_image(source_directory, output_directory_color, mode='color')