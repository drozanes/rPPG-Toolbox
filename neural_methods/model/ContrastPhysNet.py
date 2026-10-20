import torch
import torch.nn as nn
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX

class ContrastPhysNet(PhysNet_padding_Encoder_Decoder_MAX):
    """
    ContrastPhysNet: 3D-CNN architecture for Contrast-Phys+ (TPAMI 2024 / ECCV 2022).
    Outputs a Spatio-Temporal rPPG (ST-rPPG) block with S x S spatial resolution (default 2x2 = 4 quadrants)
    and the spatially averaged rPPG signal for evaluation.
    """
    def __init__(self, frames=128, S=2, use_fusion_stem: bool = False, fusion_stem_alpha: float = 0.5, fusion_stem_beta: float = 0.5):
        super(ContrastPhysNet, self).__init__(
            frames=frames,
            use_fusion_stem=use_fusion_stem,
            fusion_stem_alpha=fusion_stem_alpha,
            fusion_stem_beta=fusion_stem_beta
        )
        self.S = S
        self.poolspa_contrast = nn.AdaptiveAvgPool3d((None, S, S))
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
    
    def forward(self, x):
        # x: [Batch, 3, T, 128, 128]
        # Instance normalization per video clip to remove illumination offsets
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / (stds + 1e-6)

        [batch, channel, length, width, height] = x.shape

        if self.use_fusion_stem:
            x_fused = self.fusion_stem(x)
            x = self.temporal_pool(x_fused)
        else:
            x = self.ConvBlock1(x)  # [16, T, 128, 128]
            x = self.MaxpoolSpa(x)  # [16, T, 64, 64]

            x = self.ConvBlock2(x)  # [32, T, 64, 64]
            x_visual6464 = self.ConvBlock3(x)  # [64, T, 64, 64]
            x = self.MaxpoolSpaTem(x_visual6464) # [64, T/2, 32, 32]

        x = self.ConvBlock4(x)  # [64, T/2, 32, 32]
        x_visual3232 = self.ConvBlock5(x)  # [64, T/2, 32, 32]
        x = self.MaxpoolSpaTem(x_visual3232)  # [64, T/4, 16, 16]

        x = self.ConvBlock6(x)  # [64, T/4, 16, 16]
        x_visual1616 = self.ConvBlock7(x)  # [64, T/4, 16, 16]
        x = self.MaxpoolSpa(x_visual1616)  # [64, T/4, 8, 8]

        x = self.ConvBlock8(x)  # [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # [64, T/4, 8, 8]
        x = self.upsample(x)  # [64, T/2, 8, 8]
        x = self.upsample2(x)  # [64, T, 8, 8]

        # Contrast-Phys+ Spatiotemporal rPPG Block:
        # Pool spatially to S x S quadrants (e.g. 2x2 = 4 quadrants: forehead, cheeks, chin)
        x_st = self.poolspa_contrast(x)  # [B, 64, T, S, S]
        st_block = self.ConvBlock10(x_st)  # [B, 1, T, S, S]

        # Extract spatial quadrant signals
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(st_block[:, :, :, a, b])  # [B, 1, T]

        # Spatial average across all quadrants: [B, 1, T] -> [B, T]
        avg_pulse = sum(x_list) / (self.S * self.S)
        rPPG = avg_pulse.squeeze(1)  # [B, T]

        # Flattened spatial representation for contrastive sampling: [B, S*S, T] = [B, 4, T]
        st_signals = torch.cat(x_list, dim=1)  # [B, S*S, T]

        return rPPG, st_signals, x_visual3232, x_visual1616
