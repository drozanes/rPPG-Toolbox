"""Fusion Stem Module for rPPG Networks.

Introduced in:
"RhythmFormer: Extracting Patterned rPPG Signals based on Periodic Sparse Attention"
(CVPR 2024 / arXiv:2402.12788)
Authors: Bochao Zou, Zizheng Guo, Jiansheng Chen, Junbao Zhuo, Weiran Huang, Huimin Ma

The Fusion Stem integrates temporal context into video-frame representations by combining
raw appearance frames (X_raw) with multi-frame temporal differences (X_diff).
This guides feature extraction and self-attention mechanisms toward blood volume pulse (BVP)
variations while filtering out background noise and motion artifacts.

This implementation is modular, plug-and-play, and supports:
- 5D video inputs: [N, D, C, H, W] or [N, C, D, H, W]
- 4D frame inputs: [B, C, H, W] (e.g. C=6 with diff+raw or C=3)
- Flexible output shapes: 'NCDHW', 'NDCHW', or 'ND_flattened' ([N*D, C_out, H', W'])
- Configurable alpha/beta fusion weights and channel widths
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn


class FusionStem(nn.Module):
    """Modular, plug-and-play Fusion Stem from RhythmFormer (CVPR 2024).

    Constructs 4 consecutive temporal differences around each frame (t-2, t-1, t, t+1, t+2),
    processes raw and difference streams in parallel, and applies a two-stage hierarchical fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        alpha: float = 0.5,
        beta: float = 0.5,
        downsample_stride: int = 2,
        output_format: str = "NCDHW",
    ):
        """
        Args:
            in_channels: Number of color channels in raw frame (default: 3).
            out_channels: Number of feature channels after the stem (default: 64).
            alpha: Weight for raw appearance branch in fusion (default: 0.5).
            beta: Weight for temporal difference branch in fusion (default: 0.5).
            downsample_stride: Spatial stride in stem convs (default: 2, resulting in 4x total downsampling with maxpool).
            output_format: Shape of output tensor for 5D inputs.
                - 'NCDHW': [N, out_channels, D, H//4, W//4] (ideal for 3D CNNs / 3D ViTs like PhysNet, PhysFormer)
                - 'NDCHW': [N, D, out_channels, H//4, W//4] (ideal for temporal sequence models)
                - 'ND_flattened': [N*D, out_channels, H//4, W//4] (used by RhythmFormer & 2D frame models)
        """
        super(FusionStem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.downsample_stride = downsample_stride
        self.output_format = output_format

        # Stem 11: Raw appearance branch (takes 3 channels)
        if downsample_stride == 2:
            self.stem11 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
            # Stem 12: Multi-temporal difference branch (takes 4 x in_channels = 12 channels)
            self.stem12 = nn.Sequential(
                nn.Conv2d(in_channels * 4, out_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        else:
            self.stem11 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
            self.stem12 = nn.Sequential(
                nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Stage 1 fusion refinement
        self.stem21 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 2 fusion refinement
        self.stem22 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Fusion Stem.

        Args:
            x: Input tensor. Supported shapes:
               - [N, D, C, H, W] (5D video with C=3 at dim 2)
               - [N, C, D, H, W] (5D video with C=3 at dim 1)
               - [B, C, H, W] (4D frame batch, e.g. C=6 with [diff, raw] or C=3 raw)
        Returns:
            fused_x: Feature tensor with out_channels. Shape corresponds to output_format.
        """
        if x.ndim == 5:
            return self._forward_5d(x)
        elif x.ndim == 4:
            return self._forward_4d(x)
        else:
            raise ValueError(f"FusionStem expects a 4D or 5D tensor, but received tensor of shape {x.shape}")

    def _forward_5d(self, x: torch.Tensor) -> torch.Tensor:
        # Detect channel ordering: [N, C, D, H, W] vs [N, D, C, H, W]
        if x.shape[1] == self.in_channels and x.shape[2] != self.in_channels:
            # [N, C, D, H, W] -> permute to [N, D, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)

        N, D, C, H, W = x.shape

        # Construct 5 consecutive temporal shifts with edge reflection/clamping:
        # x1 = x_{t-2}, x2 = x_{t-1}, x3 = x_t (center), x4 = x_{t+1}, x5 = x_{t+2}
        if D >= 3:
            x1 = torch.cat([x[:, :1, :, :, :], x[:, :1, :, :, :], x[:, :D-2, :, :, :]], dim=1)
            x2 = torch.cat([x[:, :1, :, :, :], x[:, :D-1, :, :, :]], dim=1)
            x3 = x
            x4 = torch.cat([x[:, 1:, :, :, :], x[:, D-1:, :, :, :]], dim=1)
            x5 = torch.cat([x[:, 2:, :, :, :], x[:, D-1:, :, :, :], x[:, D-1:, :, :, :]], dim=1)
        else:
            # Fallback for very short clips: replicate center frame
            x1 = x
            x2 = x
            x3 = x
            x4 = x
            x5 = x

        # 4 consecutive temporal differences concatenated along channel axis -> [N, D, 12, H, W]
        diff_seq = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], dim=2)
        diff_seq = diff_seq.contiguous().view(N * D, C * 4, H, W)

        # Raw center frames -> [N*D, C, H, W]
        raw_seq = x3.contiguous().view(N * D, C, H, W)

        # Stream feature extraction
        x_diff = self.stem12(diff_seq)
        x_raw = self.stem11(raw_seq)

        # Hierarchical two-stage fusion (Eqs. in RhythmFormer CVPR 2024 Sec. 3.2)
        # Stage 1:
        x_path1 = self.alpha * x_raw + self.beta * x_diff
        x_path1 = self.stem21(x_path1)

        # Stage 2:
        x_path2 = self.stem22(x_diff)
        x_fused = self.alpha * x_path1 + self.beta * x_path2

        # Format output
        _, C_out, H_out, W_out = x_fused.shape
        if self.output_format == "NCDHW":
            return x_fused.view(N, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        elif self.output_format == "NDCHW":
            return x_fused.view(N, D, C_out, H_out, W_out)
        elif self.output_format in ("ND_flattened", "flattened"):
            return x_fused
        else:
            raise ValueError(f"Unknown output_format: {self.output_format}")

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C == 6:
            # 2D frame difference models (e.g. DeepPhys): diff=x[:, :3], raw=x[:, 3:]
            diff = x[:, :3, :, :]
            raw = x[:, 3:, :, :]
            # Repeat diff 4x to match the 12-channel diff input expected by stem12
            diff_multi = diff.repeat(1, 4, 1, 1)
        elif C == 12:
            diff_multi = x
            raw = x[:, :3, :, :]
        elif C == 3:
            raw = x
            diff_multi = torch.zeros(B, 12, H, W, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"FusionStem 4D expects C=3, 6, or 12, got C={C}")

        x_diff = self.stem12(diff_multi)
        x_raw = self.stem11(raw)

        x_path1 = self.alpha * x_raw + self.beta * x_diff
        x_path1 = self.stem21(x_path1)
        x_path2 = self.stem22(x_diff)
        x_fused = self.alpha * x_path1 + self.beta * x_path2

        return x_fused


# Alias for backward compatibility with RhythmFormer
Fusion_Stem = FusionStem
