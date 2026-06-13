import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, load_model, parse_devices
from utils.visualize import print_iou
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

logger = get_logger()


# -----------------------------------------------------------------------------
# Palette utilities
# -----------------------------------------------------------------------------
def get_safe_name(name):
    if isinstance(name, (list, tuple)):
        name = name[0]
    name = os.path.splitext(str(name))[0]
    return name


# 参考你给出的示例图，总体是“深蓝背景 + 青绿/浅蓝 + 金黄 + 橙红”这一类高对比配色。
# 下面这套 palette 不是逐像素抠出来的“完全同款”，而是更适合论文可视化的近似复现版。
# 若类别数超过当前长度，会自动循环补足。
REFERENCE_PALETTE_RGB = [
    (0, 0, 255),        # 0: 深蓝背景
    (113, 208, 177),    # 1: 青绿色
    (125, 171, 222),    # 2: 浅蓝
    (210, 88, 18),      # 3: 橙红
    (235, 198, 40),     # 4: 金黄
    (245, 164, 32),     # 5: 橙黄
    (160, 220, 170),    # 6: 浅绿
    (145, 186, 218),    # 7: 淡蓝灰
    (255, 255, 255),    # 8: 白色高亮
    (188, 118, 232),    # 9: 紫色
    (89, 220, 236),     # 10: 亮青
    (255, 126, 95),     # 11: 珊瑚橙
    (214, 225, 76),     # 12: 黄绿
    (170, 170, 170),    # 13: 中灰
    (95, 235, 180),     # 14: 绿色青
    (255, 214, 102),    # 15: 浅金
    (118, 150, 255),    # 16: 亮蓝
    (255, 160, 160),    # 17: 浅红
    (195, 255, 170),    # 18: 浅黄绿
]


def get_reference_palette(num_classes):
    palette = []
    for i in range(num_classes):
        palette.append(REFERENCE_PALETTE_RGB[i % len(REFERENCE_PALETTE_RGB)])
    return palette


def get_class_colors_from_dataset(dataset):
    """
    兼容 dataset.get_class_colors 是函数或属性两种情况
    """
    if hasattr(dataset, "get_class_colors"):
        if callable(dataset.get_class_colors):
            return dataset.get_class_colors()
        return dataset.get_class_colors
    raise AttributeError("Dataset does not provide get_class_colors")


def get_class_colors(dataset, num_classes, palette_style="reference"):
    """
    palette_style:
        - dataset   : 使用数据集原生 palette
        - reference : 使用你给出的参考图风格 palette
    """
    if palette_style == "dataset":
        return get_class_colors_from_dataset(dataset)
    if palette_style == "reference":
        return get_reference_palette(num_classes)
    raise ValueError(f"Unsupported palette_style: {palette_style}")


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def label_to_color_image(label, class_colors):
    """
    label: H x W
    return: H x W x 3, uint8
    """
    h, w = label.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(class_colors):
        color_img[label == cls_id] = color
    return color_img


def recover_vis_image(img, mean=None, std=None, is_single_channel=False):
    """
    将 dataloader 输出尽量恢复成可视化图像
    支持输入：
    - torch.Tensor 或 np.ndarray
    - CHW 或 HWC
    - 单通道 / 三通道

    注意：
    1. 如果你的输入已经是 uint8 原图，这里会直接返回
    2. 如果你的输入是 normalize 后的 float，会尝试反归一化
    3. 对 X / R 模态，若没有明确 mean/std，可传 None，函数会做 min-max 可视化
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    img = np.array(img)

    while img.ndim > 3:
        img = np.squeeze(img, axis=0)

    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        if img.dtype == np.uint8:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
        if img.dtype == np.uint8:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype == np.uint8:
            return img

        if mean is not None and std is not None:
            mean = np.array(mean).reshape(1, 1, 3)
            std = np.array(std).reshape(1, 1, 3)
            img = img * std + mean

        if img.max() <= 1.5:
            img = img * 255.0

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    raise ValueError(f"Unsupported image shape for visualization: {img.shape}")


def make_overlay(image_bgr, pred, class_colors, alpha=0.55):
    color_mask_rgb = label_to_color_image(pred, class_colors)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, color_mask_bgr, alpha, 0)
    return overlay


def put_title(img, title, font_scale=0.8, thickness=2):
    h, w = img.shape[:2]
    bar_h = 36
    canvas = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 255
    canvas[bar_h:] = img
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def resize_to_same_height(img_list, target_h=None):
    if target_h is None:
        target_h = min([img.shape[0] for img in img_list])
    out = []
    for img in img_list:
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        out.append(cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR))
    return out


def make_paper_panel(rgb_img, x_img, r_img, gt_color, pred_color, overlay_img):
    row1 = [put_title(rgb_img, "RGB"),
            put_title(x_img, "X"),
            put_title(r_img, "R")]
    row2 = [put_title(gt_color, "GT"),
            put_title(pred_color, "Pred"),
            put_title(overlay_img, "Overlay")]

    row1 = resize_to_same_height(row1, target_h=320)
    row2 = resize_to_same_height(row2, target_h=320)

    row1 = cv2.hconcat(row1)
    row2 = cv2.hconcat(row2)

    panel_w = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] != panel_w:
        row1 = cv2.copyMakeBorder(row1, 0, 0, 0, panel_w - row1.shape[1],
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if row2.shape[1] != panel_w:
        row2 = cv2.copyMakeBorder(row2, 0, 0, 0, panel_w - row2.shape[1],
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    panel = cv2.vconcat([row1, row2])
    return panel


class SegEvaluator(Evaluator):
    def __init__(self, *args, paper_vis=False, palette_style="reference", overlay_alpha=0.55, **kwargs):
        super().__init__(*args, **kwargs)
        self.paper_vis = paper_vis
        self.palette_style = palette_style
        self.overlay_alpha = overlay_alpha

    def save_visualizations(self, data, pred):
        if self.save_path is None:
            return

        save_root = self.save_path
        raw_dir = os.path.join(save_root, "pred_raw")
        color_dir = os.path.join(save_root, "pred_color")
        overlay_dir = os.path.join(save_root, "overlay")
        panel_dir = os.path.join(save_root, "paper_panel")

        ensure_dir(save_root)
        ensure_dir(raw_dir)
        ensure_dir(color_dir)
        ensure_dir(overlay_dir)
        ensure_dir(panel_dir)

        name = get_safe_name(data["fn"])
        fn = name + ".png"

        img = data["data"]
        label = data["label"]
        modal_x = data["modal_x"]
        modal_r = data["modal_r"]

        class_colors = get_class_colors(self.dataset, config.num_classes, self.palette_style)

        cv2.imwrite(os.path.join(raw_dir, fn), pred.astype(np.uint8))

        pred_pil = Image.fromarray(pred.astype(np.uint8), mode="P")
        palette_list = list(np.array(class_colors).reshape(-1))
        if len(palette_list) < 768:
            palette_list += [0] * (768 - len(palette_list))
        pred_pil.putpalette(palette_list)
        pred_pil.save(os.path.join(color_dir, fn))

        rgb_img = recover_vis_image(
            img,
            mean=config.norm_mean,
            std=config.norm_std,
            is_single_channel=False
        )
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) if rgb_img.shape[2] == 3 else rgb_img

        x_img = recover_vis_image(
            modal_x,
            mean=None,
            std=None,
            is_single_channel=getattr(config, "x_is_single_channel", False)
        )
        r_img = recover_vis_image(
            modal_r,
            mean=None,
            std=None,
            is_single_channel=getattr(config, "r_is_single_channel", False)
        )

        x_bgr = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR) if x_img.ndim == 3 and x_img.shape[2] == 3 else x_img
        r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR) if r_img.ndim == 3 and r_img.shape[2] == 3 else r_img

        gt_color = label_to_color_image(label.astype(np.uint8), class_colors)
        pred_color = label_to_color_image(pred.astype(np.uint8), class_colors)

        gt_bgr = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(rgb_bgr, 1 - self.overlay_alpha, pred_bgr, self.overlay_alpha, 0)
        cv2.imwrite(os.path.join(overlay_dir, fn), overlay)

        if self.paper_vis:
            panel = make_paper_panel(
                rgb_bgr, x_bgr, r_bgr,
                gt_bgr, pred_bgr, overlay
            )
            cv2.imwrite(os.path.join(panel_dir, fn), panel)

        logger.info(f"Saved visualization: {fn}")

    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        modal_r = data['modal_r']

        pred = self.sliding_eval_rgbX(
            img, modal_x, modal_r,
            config.eval_crop_size,
            config.eval_stride_rate,
            device
        )

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {
            'hist': hist_tmp,
            'labeled': labeled_tmp,
            'correct': correct_tmp
        }

        self.save_visualizations(data, pred)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0

        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
            hist, correct, labeled
        )

        result_line = print_iou(
            iou, freq_IoU, mean_pixel_acc, pixel_acc,
            self.dataset.class_names, show_no_back=False
        )
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='300', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default='./vis_stsd300')
    parser.add_argument('--paper_vis', action='store_true',
                        help='whether to save paper-style concatenated panels')
    parser.add_argument('--palette_style', default='reference', choices=['reference', 'dataset'],
                        help='reference: 使用参考图风格配色; dataset: 使用数据集原生配色')
    parser.add_argument('--overlay_alpha', default=0.55, type=float,
                        help='overlay 中分割 mask 的透明度')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)

    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'x_root': config.x_root_folder,
        'x_format': config.x_format,
        'x_single_channel': config.x_is_single_channel,
        'r_root': config.r_root_folder,
        'r_format': config.r_format,
        'r_single_channel': config.r_is_single_channel,
        'class_names': config.class_names,
        'train_source': config.train_source,
        'eval_source': config.eval_source,
    }

    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(
            dataset,
            config.num_classes,
            config.norm_mean,
            config.norm_std,
            network,
            config.eval_scale_array,
            config.eval_flip,
            all_dev,
            args.verbose,
            args.save_path,
            False,
            paper_vis=args.paper_vis,
            palette_style=args.palette_style,
            overlay_alpha=args.overlay_alpha,
        )
        segmentor.run(
            config.checkpoint_dir,
            args.epochs,
            config.val_log_file,
            config.link_val_log_file
        )
