import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
#from mmcv.cnn import build_norm_layer
from engine.logger import get_logger
from mmcv.cnn import build_norm_layer
logger = get_logger()


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class DWConv1(nn.Module):
    def __init__(self, dim=768):
        super(DWConv1, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv1(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):


        x = self.fc1(x)

        x = self.dwconv(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)


        return x

'''
class EdgeEnhanceModule(nn.Module):
    """边缘增强模块 - 提取深度/热力图的边缘信息"""

    def __init__(self, dim):
        super().__init__()
        self.edge_conv_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.edge_conv_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        for i in range(self.edge_conv_x.weight.shape[0]):
            self.edge_conv_x.weight.data[i, 0] = sobel_x / 8.0
            self.edge_conv_y.weight.data[i, 0] = sobel_y / 8.0

    def forward(self, x):
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)
        edge_feat = self.edge_fusion(torch.cat([edge_x, edge_y], dim=1))
        return edge_feat


class FeatureCalibration(nn.Module):
    """特征校准模块 - 自适应调整特征响应"""

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        combined = torch.cat([avg_out, max_out], dim=1)
        weight = self.fc(combined)
        return x * weight


class EnhancedMSPoolAttention(nn.Module):
    """改进的多尺度池化注意力 - 针对灰度图优化"""

    def __init__(self, dim):
        super().__init__()
        pools = [3, 5, 7, 11]
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
        self.pool4 = nn.AvgPool2d(pools[3], stride=1, padding=pools[3] // 2, count_include_pad=False)
        self.scale_weights = nn.Parameter(torch.ones(5) / 5)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.conv_out = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        x_in = self.conv0(x)
        x_1 = self.pool1(x_in)
        x_2 = self.pool2(x_in)
        x_3 = self.pool3(x_in)
        x_4 = self.pool4(x_in)
        weights = F.softmax(self.scale_weights, dim=0)
        x_multi = (weights[0] * x_in + weights[1] * x_1 +
                   weights[2] * x_2 + weights[3] * x_3 + weights[4] * x_4)
        x_fused = self.fusion_conv(x_multi)
        x_out = self.sigmoid(self.conv_out(x_fused)) * u
        return x_out + u


class MSPABlock(nn.Module):
    """增强版MSPA Block - 专门优化灰度图像特征提取"""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = EnhancedMSPoolAttention(dim)  # 使用增强版注意力
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        # 新增模块
        self.edge_enhance = EdgeEnhanceModule(dim)
        self.edge_norm = nn.BatchNorm2d(dim)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.edge_weight = nn.Parameter(torch.tensor(0.05))  # 初始更小
        self.edge_max = 0.2  # 强度上限（避免训练后期变太强）
        self.feature_calibration = FeatureCalibration(dim, reduction=16)


        self.is_channel_mix = False
        if self.is_channel_mix:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.c_nets = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        # 先做 attention
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))

        # ✅ edge enhance
        edge_feat = self.edge_enhance(x)
        edge_feat = self.edge_norm(edge_feat)

        # ✅ 边缘 mask（只在边缘区域增强）
        edge_mag = edge_feat.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        edge_mask = torch.sigmoid(5.0 * (edge_mag - edge_mag.mean()))  # soft mask

        # ✅ gate（网络学习在哪些通道/位置需要增强）
        gate = self.edge_gate(x)

        # ✅ 限制 weight 范围，避免过强
        w = torch.clamp(torch.abs(self.edge_weight), 0.0, self.edge_max)

        x = x + w * gate * edge_mask * edge_feat

        # MLP
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))

        # calibration
        x = self.feature_calibration(x)
        return x
        '''
'''
    def forward(self, x):
        edge_feat = self.edge_enhance(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + torch.abs(self.edge_weight) * edge_feat

        if self.is_channel_mix:
            x_c = self.avg_pool(x)
            x_c = self.c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            x_c = x_c.expand_as(x)
            x_c_mix = x_c * x
            x_mlp = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
            x = x_c_mix + x_mlp
        else:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))

        x = self.feature_calibration(x)
        return x
        '''
'''
class MSPoolAttentionV2(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        pools = [3, 5, 7]
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0]//2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1]//2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2]//2, count_include_pad=False)

        # ✅ learnable scale weights
        self.scale = nn.Parameter(torch.ones(4))  # x_in + pool1 + pool2 + pool3

        # ✅ channel gate (SE)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # ✅ stronger spatial gate head
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x
        x_in = self.conv0(x)
        x_1 = self.pool1(x_in)
        x_2 = self.pool2(x_in)
        x_3 = self.pool3(x_in)

        w = F.softmax(self.scale, dim=0)
        x_fused = w[0]*x_in + w[1]*x_1 + w[2]*x_2 + w[3]*x_3

        # channel refine
        x_fused = x_fused * self.se(x_fused)

        # spatial gate
        gate = self.spatial_gate(x_fused)
        out = gate * u
        return out + u
class MSPABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = MSPoolAttentionV2(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.is_channel_mix = False
        if self.is_channel_mix:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.c_nets = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))

        if self.is_channel_mix:
            x_c = self.avg_pool(x)
            x_c = self.c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            x_c = x_c.expand_as(x)
            x_c_mix = x_c * x
            x_mlp = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
            x = x_c_mix + x_mlp
        else:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
'''
class MSPoolAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        pools = [3, 5, 7]
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        x_in = self.conv0(x)
        x_1 = self.pool1(x_in)
        x_2 = self.pool2(x_in)
        x_3 = self.pool3(x_in)
        x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
        return x_out + u


class MSPABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = MSPoolAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.is_channel_mix = False
        if self.is_channel_mix:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.c_nets = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))

        if self.is_channel_mix:
            x_c = self.avg_pool(x)
            x_c = self.c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            x_c = x_c.expand_as(x)
            x_c_mix = x_c * x
            x_mlp = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
            x = x_c_mix + x_mlp
        else:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)

        return x, H, W


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        norm_cfg = dict(type='BN', requires_grad=True)
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        self.extra_r_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                      embed_dim=embed_dims[0])
        self.extra_r_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                      in_chans=embed_dims[0],
                                                      embed_dim=embed_dims[1])
        self.extra_r_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                      in_chans=embed_dims[1],
                                                      embed_dim=embed_dims[2])
        self.extra_r_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                      in_chans=embed_dims[2],
                                                      embed_dim=embed_dims[3])
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([MSPABlock(embed_dims[0], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[0])])
        self.extra_norm1 = ConvLayerNorm(embed_dims[0])

        self.extra_r_block1 = nn.ModuleList([MSPABlock(embed_dims[0], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[0])])
        self.extra_r_norm1 = ConvLayerNorm(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([MSPABlock(embed_dims[1], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[1])])
        self.extra_norm2 = ConvLayerNorm(embed_dims[1])

        self.extra_r_block2 = nn.ModuleList([MSPABlock(embed_dims[1], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[1])])
        self.extra_r_norm2 = ConvLayerNorm(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[2])])
        self.extra_norm3 = ConvLayerNorm(embed_dims[2])

        self.extra_r_block3 = nn.ModuleList([MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[2])])
        self.extra_r_norm3 = ConvLayerNorm(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[3])])
        self.extra_norm4 = ConvLayerNorm(embed_dims[3])

        self.extra_r_block4 = nn.ModuleList([MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg)
            for i in range(depths[3])])
        self.extra_r_norm4 = ConvLayerNorm(embed_dims[3])
        cur += depths[3]

        self.FRMs = nn.ModuleList([
            FRM(dim=embed_dims[0], reduction=1),
            FRM(dim=embed_dims[1], reduction=1),
            FRM(dim=embed_dims[2], reduction=1),
            FRM(dim=embed_dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
            FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
            FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
            FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb, x_e, x_e1):
        """
        x_rgb: B x N x H x W
        """
        B = x_rgb.shape[0]
        outs = []
        outs_fused = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # B H*W/16 C
        x_e, _, _ = self.extra_patch_embed1(x_e)
        x_e1, _, _ = self.extra_r_patch_embed1(x_e1)

        for i, blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e1 = x_e1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        for i, blk in enumerate(self.extra_block1):
            x_e = blk(x_e)
        for i, blk in enumerate(self.extra_r_block1):
            x_e1 = blk(x_e1)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)
        x_e1 = self.extra_r_norm1(x_e1)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb, x_e, x_e1 = self.FRMs[0](x_rgb, x_e, x_e1)
        x_fused = self.FFMs[0](x_rgb, x_e, x_e1)
        outs.append(x_fused)

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        x_e1, _, _ = self.extra_r_patch_embed2(x_e1)

        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e1 = x_e1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        for i, blk in enumerate(self.extra_block2):
            x_e = blk(x_e)
        for i, blk in enumerate(self.extra_r_block2):
            x_e1 = blk(x_e1)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)
        x_e1 = self.extra_r_norm2(x_e1)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb, x_e, x_e1 = self.FRMs[1](x_rgb, x_e, x_e1)
        x_fused = self.FFMs[1](x_rgb, x_e, x_e1)
        outs.append(x_fused)

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        x_e1, _, _ = self.extra_r_patch_embed3(x_e1)

        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e1 = x_e1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        for i, blk in enumerate(self.extra_block3):
            x_e = blk(x_e)
        for i, blk in enumerate(self.extra_r_block3):
            x_e1 = blk(x_e1)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)
        x_e1 = self.extra_r_norm3(x_e1)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb, x_e, x_e1 = self.FRMs[2](x_rgb, x_e, x_e1)
        x_fused = self.FFMs[2](x_rgb, x_e, x_e1)
        outs.append(x_fused)

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        x_e1, _, _ = self.extra_r_patch_embed4(x_e1)

        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e1 = x_e1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        for i, blk in enumerate(self.extra_block4):
            x_e = blk(x_e)
        for i, blk in enumerate(self.extra_r_block4):
            x_e1 = blk(x_e1)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)
        x_e1 = self.extra_r_norm4(x_e1)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb, x_e, x_e1 = self.FRMs[3](x_rgb, x_e, x_e1)
        x_fused = self.FFMs[3](x_rgb, x_e, x_e1)
        outs.append(x_fused)

        return outs

    def forward(self, x_rgb, x_e, x_e1):
        out = self.forward_features(x_rgb, x_e, x_e1)
        return out


'''def load_dualpath_model(model, model_file):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        # raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
'''


def load_dualpath_model(model, model_file):
    """
    加载预训练模型，并自动处理维度不匹配问题
    """
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
            state_dict[k.replace('patch_embed', 'extra_r_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v

            # 对于 extra_block，需要转换 MLP 权重的形状
            extra_key = k.replace('block', 'extra_block')
            extra_r_key = k.replace('block', 'extra_r_block')

            # 如果是 mlp.fc1 或 mlp.fc2 的权重，需要从 2D 转为 4D
            if 'mlp.fc1.weight' in k or 'mlp.fc2.weight' in k:
                # 原始权重: (out_channels, in_channels)
                # Conv2d 需要: (out_channels, in_channels, 1, 1)
                v_conv = v.unsqueeze(-1).unsqueeze(-1)
                state_dict[extra_key] = v_conv
                state_dict[extra_r_key] = v_conv
            # 如果是 mlp.fc1 或 mlp.fc2 的 bias，直接复制
            elif 'mlp.fc1.bias' in k or 'mlp.fc2.bias' in k:
                state_dict[extra_key] = v
                state_dict[extra_r_key] = v
            # 其他参数（如 dwconv）直接复制
            else:
                state_dict[extra_key] = v
                state_dict[extra_r_key] = v

        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v
            state_dict[k.replace('norm', 'extra_r_norm')] = v

    t_ioend = time.time()

    # 加载权重，允许部分不匹配（strict=False）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # 打印加载信息
    if missing_keys:
        logger.info(f"Missing keys in state_dict: {len(missing_keys)} keys")
        # 只打印前5个，避免输出太长
        logger.info(f"First few missing keys: {missing_keys[:5]}")

    if unexpected_keys:
        logger.info(f"Unexpected keys in state_dict: {len(unexpected_keys)} keys")
        logger.info(f"First few unexpected keys: {unexpected_keys[:5]}")

    del state_dict

    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)