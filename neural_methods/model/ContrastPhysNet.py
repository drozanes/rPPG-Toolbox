import torch
import torch.nn as nn
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX

class ContrastPhysNet(PhysNet_padding_Encoder_Decoder_MAX):
    def __init__(self, frames=128):
        super(ContrastPhysNet, self).__init__(frames=frames)
        # Remove the AdaptiveAvgPool3d and redefine ConvBlock10 if needed, 
        # but actually we can just override forward.
    
    def forward(self, x):
        # x: [Batch, 3, T, 128, 128]
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [16, T, 128, 128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64, 64]

        x = self.ConvBlock2(x)  # x [32, T, 64, 64]
        x_visual6464 = self.ConvBlock3(x)  # x [64, T, 64, 64]
        x = self.MaxpoolSpaTem(x_visual6464) # x [64, T/2, 32, 32]

        x = self.ConvBlock4(x)  # x [64, T/2, 32, 32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32, 32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16, 16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16, 16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16, 16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8, 8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # Contrast-Phys+ uses spatiotemporal block
        # Instead of pooling spatially, we keep the spatial dimensions
        st_block = self.ConvBlock10(x) # [B, 1, T, 8, 8]
        
        # During inference/standard evaluation, we might want the averaged signal
        # but during training we need the block.
        # Let's return both or just the block and let the trainer handle it.
        
        # Flatten spatial dimensions: [B, 1, T, 8, 8] -> [B, T, 64] -> [B, 64, T]
        B, C, T, H, W = st_block.shape
        st_block = st_block.view(B, T, H * W).permute(0, 2, 1) # [B, 64, T]
        
        # For evaluation, we usually average
        rPPG = torch.mean(st_block, dim=1) # [B, T]

        return rPPG, st_block, x_visual3232, x_visual1616
