"""ContrastPhysNet: 3D-CNN architecture for Contrast-Phys+ (TPAMI 2024 / ECCV 2022).
Official paper:
"Contrast-Phys+: Unsupervised and Weakly-Supervised Video-Based Remote Physiological
Measurement via Spatiotemporal Contrast", IEEE TPAMI 2024.
Authors: Zhaodong Sun and Xiaobai Li.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_methods.model.FusionStem import FusionStem


class ContrastPhysNet(nn.Module):
    """Official 3D-CNN architecture for Contrast-Phys+ (TPAMI 2024).

    Outputs an ST-rPPG block with S x S spatial resolution (default 2x2 = 4 quadrants)
    and the spatially averaged rPPG signal for evaluation.
    """
    def __init__(
        self,
        frames: int = 128,
        S: int = 2,
        in_ch: int = 3,
        use_fusion_stem: bool = False,
        fusion_stem_alpha: float = 0.5,
        fusion_stem_beta: float = 0.5
    ):
        super(ContrastPhysNet, self).__init__()
        self.S = S
        self.frames = frames
        self.use_fusion_stem = use_fusion_stem

        if self.use_fusion_stem:
            self.fusion_stem = FusionStem(
                in_channels=in_ch,
                out_channels=32,
                alpha=fusion_stem_alpha,
                beta=fusion_stem_beta,
                downsample_stride=1,
                output_format="NCDHW"
            )
            start_in_ch = 32
        else:
            self.fusion_stem = None
            start_in_ch = in_ch

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=start_in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # Loop 1 (1x temporal, 2x spatial downsampling)
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # Encoder 1 (2x temporal, 2x spatial downsampling)
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # Encoder 2 (2x temporal, 2x spatial downsampling)
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # Loop 4 (1x temporal, 2x spatial downsampling -> 8x8 spatial)
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # Decoders to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.conv_end = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

    def forward(self, x):
        # x: [Batch, C, T, H, W]
        if self.use_fusion_stem:
            x = self.fusion_stem(x)
        else:
            means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
            stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
            x = (x - means) / (stds + 1e-6)

        parity = []
        x = self.start(x)
        x = self.loop1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        x = F.interpolate(x, scale_factor=(2, 1, 1), mode='nearest')
        x = self.decoder1(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1), mode='nearest')
        x = self.decoder2(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')

        # Pool spatially to S x S quadrants (e.g. 2x2 = 4 quadrants)
        if x.shape[-2] % self.S == 0 and x.shape[-1] % self.S == 0:
            k_h = x.shape[-2] // self.S
            k_w = x.shape[-1] // self.S
            x_st = F.avg_pool3d(x, kernel_size=(1, k_h, k_w), stride=(1, k_h, k_w))
        else:
            B_c, C_c, T_c, H_c, W_c = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(B_c * T_c, C_c, H_c, W_c)
            x_st = F.adaptive_avg_pool2d(x_2d, (self.S, self.S)).view(B_c, T_c, C_c, self.S, self.S).permute(0, 2, 1, 3, 4)

        x_st = self.conv_end(x_st)  # (B, 1, T, S, S)

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x_st[:, 0, :, a, b])  # (B, T)

        rppg = sum(x_list) / (self.S * self.S)  # (B, T)
        st_signals = torch.stack(x_list, dim=1)  # (B, S*S, T)

        # X is (B, S*S + 1, T) where [:, -1] is the averaged rppg
        X = torch.cat([st_signals, rppg.unsqueeze(1)], dim=1)

        # Return rppg as first output (for test/inference), st_signals for loss, and dummy visuals
        return rppg, st_signals, X, None
