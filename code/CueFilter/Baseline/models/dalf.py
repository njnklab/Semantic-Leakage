import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 简单工具：激活与初始化
def _conv1d_same_padding(kernel_size: int, dilation: int = 1) -> int:
    # 计算same padding（对称）
    return ((kernel_size - 1) * dilation) // 2


@dataclass
class DFBLConfig:
    sample_rate: int = 16000                 # 采样率
    num_filters: int = 64                    # 滤波器数量
    gabor_kernel: int = 401                  # Gabor核长度
    gabor_stride: int = 1                    # Gabor卷积步幅
    pool_kernel: int = 401                   # Kaiser池化核
    pool_stride: int = 160                   # 降采样步幅（≈10ms）
    pcen_smooth: float = 0.025               # PCEN 平滑系数
    fmin: float = 30.0                       # 低频限制（Hz）
    fmax_frac: float = 0.5                   # 相对Nyquist上限（0.5->Nyquist）


class DFBL(nn.Module):
    """可学习时域滤波器组（Gabor近似）+ Kaiser池化 + PCEN压缩"""

    def __init__(self, cfg: DFBLConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("time_axis", self._build_time_axis(cfg.gabor_kernel, cfg.sample_rate))

        # 频率参数：学习 f_low, f_high（通过无界参数映射保证有序与边界）
        # 初始化：在 [fmin, 0.5*sr*fmax_frac] 均匀铺设，低频更密集（对数）
        fmin = cfg.fmin
        fmax = (cfg.sample_rate / 2.0) * cfg.fmax_frac
        # 对数空间初始化中心与带宽
        centers = torch.logspace(math.log10(fmin + 1e-3), math.log10(fmax), steps=cfg.num_filters)
        bandwidths = torch.full((cfg.num_filters,), 120.0)  # 初始带宽（Hz）

        # 无界参数（学习）
        self._p_center = nn.Parameter(torch.log(centers))
        self._p_bw = nn.Parameter(torch.log(torch.exp(bandwidths / 1000.0) - 1.0))

        # Kaiser beta 参数：每通道独立
        self._beta = nn.Parameter(torch.full((cfg.num_filters,), 8.0))

        # PCEN参数（每通道）：g, o, e
        self._pcen_g = nn.Parameter(torch.full((cfg.num_filters,), 0.98))
        self._pcen_o = nn.Parameter(torch.full((cfg.num_filters,), 2.0))
        self._pcen_e = nn.Parameter(torch.full((cfg.num_filters,), 0.6))

    @staticmethod
    def _build_time_axis(kernel_size: int, sample_rate: int) -> torch.Tensor:
        # 对称时间轴（秒）
        half = (kernel_size - 1) // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32) / sample_rate
        return t.view(1, 1, -1)

    def _sigma_from_bandwidth(self, bw_hz: torch.Tensor) -> torch.Tensor:
        # 简化：sigma ~ 常数 / 带宽（秒），防止数值发散
        return (0.5 / (bw_hz.clamp(min=20.0)))

    def _constrain_freqs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将无界参数映射为 (f_low, f_center, f_high)
        sr = float(self.cfg.sample_rate)
        fmin = self.cfg.fmin
        fmax = (sr / 2.0) * self.cfg.fmax_frac

        f_center = torch.exp(self._p_center).clamp(min=fmin, max=fmax - 10.0)
        bw_pos = F.softplus(self._p_bw) * 1000.0  # Hz
        # 限制带宽不超过中心到边界的两倍
        max_bw = 2.0 * torch.minimum(f_center - fmin, fmax - f_center)
        bw = torch.minimum(bw_pos, max_bw.clamp(min=50.0))

        f_low = (f_center - 0.5 * bw).clamp(min=fmin)
        f_high = (f_center + 0.5 * bw).clamp(max=fmax)
        return f_low, f_center, f_high

    def _make_gabor_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 生成复Gabor核（实部/虚部），shape: [num_filters, 1, K]
        t = self.time_axis  # [1,1,K]
        f_low, f_center, f_high = self._constrain_freqs()

        # 用中心频率与带宽推导sigma（秒），再生成高斯包络
        sigma = self._sigma_from_bandwidth(f_high - f_low).view(-1, 1, 1)
        env = torch.exp(-0.5 * (t / sigma).pow(2))  # [1,1,K] 广播为 [N,1,K]

        phase = 2.0 * math.pi * f_center.view(-1, 1, 1) * t  # [N,1,K]
        real = env * torch.cos(phase)
        imag = env * torch.sin(phase)

        return real, imag  # [N,1,K]

    @staticmethod
    def _kaiser_window(window_length: int, beta: torch.Tensor, device: torch.device) -> torch.Tensor:
        # 生成Kaiser窗：每通道一条，返回 [N, 1, K]
        n = torch.arange(window_length, device=device, dtype=torch.float32)
        i0_beta = torch.i0(beta)
        arg = beta.view(-1, 1) * torch.sqrt(1.0 - ((2.0 * n / (window_length - 1.0)) - 1.0).pow(2))
        w = torch.i0(arg) / (i0_beta.view(-1, 1) + 1e-8)
        return w.view(-1, 1, window_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T] 原始波形
        return: [B, N, T_pool] 经过DFBL的通道×时间特征
        """
        B, C, T = x.shape
        assert C == 1, "DFBL 输入通道应为1 (单声道)"

        # 复Gabor卷积，实/虚两路
        real_k, imag_k = self._make_gabor_pair()  # [N,1,K]
        real_out = F.conv1d(x, real_k, stride=self.cfg.gabor_stride, padding=self.cfg.gabor_kernel // 2)
        imag_out = F.conv1d(x, imag_k, stride=self.cfg.gabor_stride, padding=self.cfg.gabor_kernel // 2)

        # 能量scalogram（eq.(6)）
        scalogram = 0.5 * (real_out.pow(2) + imag_out.pow(2))  # [B, N, T]

        # 深度可分离Kaiser池化（每通道单独）
        beta = self._beta.clamp(min=1.0, max=20.0)
        kaiser = self._kaiser_window(self.cfg.pool_kernel, beta, scalogram.device)  # [N,1,K]
        scalogram_padded = F.pad(scalogram, (self.cfg.pool_kernel // 2, self.cfg.pool_kernel // 2))
        pooled = F.conv1d(
            scalogram_padded,
            kaiser,
            stride=self.cfg.pool_stride,
            groups=self.cfg.num_filters
        )

        # PCEN风格归一化与压缩
        pcen_out = self._pcen(pooled)
        return pcen_out  # [B, N, T_pool]

    def _pcen(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T]
        s = self.cfg.pcen_smooth
        g = self._pcen_g.clamp(0.5, 0.999).view(1, -1, 1)
        o = self._pcen_o.clamp(0.0, 10.0).view(1, -1, 1)
        e = self._pcen_e.clamp(0.1, 1.0).view(1, -1, 1)

        B, N, T = x.shape
        ema = torch.zeros((B, N), device=x.device, dtype=x.dtype)
        out = []
        eps = 1e-6
        for t in range(T):
            ema = (1.0 - s) * ema + s * x[:, :, t]
            y = (x[:, :, t] / (ema + eps) + o.squeeze(2)) ** e.squeeze(2) - o.squeeze(2) ** e.squeeze(2)
            out.append(y)
        y_seq = torch.stack(out, dim=2)
        return y_seq * g  # [B, N, T]


@dataclass
class MSSAConfig:
    in_channels: int
    meb_blocks: int = 4                     # 膨胀块堆叠数
    meb_kernel: int = 3                     # MEB卷积核
    fab_kernel_list: Tuple[int, int, int] = (3, 5, 7)
    hidden_channels: int = 128


class MEBStack(nn.Module):
    """多尺度特征提取（膨胀卷积 + 1x1 Conv）"""

    def __init__(self, cfg: MSSAConfig):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = cfg.in_channels
        for i in range(cfg.meb_blocks):
            dilation = 2 ** i
            self.blocks.append(
                nn.Sequential(
                    # 深度可分离卷积（每通道时域）
                    nn.Conv1d(in_ch, in_ch, kernel_size=cfg.meb_kernel, dilation=dilation,
                              padding=_conv1d_same_padding(cfg.meb_kernel, dilation), groups=in_ch, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(in_ch, cfg.hidden_channels, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(cfg.hidden_channels, in_ch, kernel_size=1, bias=False)
                )
            )

        # 为了形成skip融合，准备一个1x1映射
        self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: [B, C, T]
        skips: List[torch.Tensor] = []
        h = x
        for blk in self.blocks:
            y = blk(h)
            h = h + y
            skips.append(h)
        return h, skips


class FABlock(nn.Module):
    """频率感知注意力（通道加权）"""

    def __init__(self, in_channels: int, kernel_list: Tuple[int, int, int] = (3, 5, 7)):
        super().__init__()
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, k, padding=_conv1d_same_padding(k), groups=in_channels, bias=False)
            for k in kernel_list
        ])
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        feats = 0
        for conv in self.depthwise_convs:
            feats = feats + conv(x)
        feats = self.act(self.bn(feats))
        # 全局时域平均 -> 通道向量
        z = feats.mean(dim=2)  # [B, C]
        a = self.fc(z)         # [B, C]
        a = F.softmax(a, dim=1)
        return x * a.unsqueeze(2)


@dataclass
class DALFConfig:
    sample_rate: int = 16000
    num_filters: int = 64
    gabor_kernel: int = 401
    pool_kernel: int = 401
    pool_stride: int = 160
    meb_blocks: int = 4
    mssa_hidden: int = 128
    head_channels: int = 128
    dropout: float = 0.2


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=_conv1d_same_padding(kernel_size), bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))
        return x + y


class DALFNet(nn.Module):
    """DALF 主网络：DFBL -> MSSA(MEB+FAB) -> 回归头"""

    def __init__(self, cfg: DALFConfig):
        super().__init__()
        self.dfbl = DFBL(DFBLConfig(
            sample_rate=cfg.sample_rate,
            num_filters=cfg.num_filters,
            gabor_kernel=cfg.gabor_kernel,
            gabor_stride=1,
            pool_kernel=cfg.pool_kernel,
            pool_stride=cfg.pool_stride,
        ))

        self.mssa = MEBStack(MSSAConfig(
            in_channels=cfg.num_filters,
            meb_blocks=cfg.meb_blocks,
            hidden_channels=cfg.mssa_hidden
        ))

        self.fa = FABlock(cfg.num_filters)

        # 头部：小型ResNet风格
        self.head_pre = nn.Sequential(
            nn.Conv1d(cfg.num_filters, cfg.head_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(cfg.head_channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualConvBlock(cfg.head_channels, 7)
        self.res2 = ResidualConvBlock(cfg.head_channels, 5)
        self.res3 = ResidualConvBlock(cfg.head_channels, 3)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.head_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x = self.dfbl(x)                    # [B, C, T']
        h, skips = self.mssa(x)              # [B, C, T'], [list]
        x_fuse = torch.stack(skips, dim=0).sum(dim=0)  # 融合skip（式16）
        x_att = self.fa(x_fuse)              # 频率注意力（式18）

        z = self.head_pre(x_att)
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        z = z.mean(dim=2)                   # 全局时域平均
        z = self.dropout(z)
        y = self.out(z)                     # [B, 1]
        return y





