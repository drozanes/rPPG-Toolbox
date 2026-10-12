import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastPhysNet(nn.Module):
    def __init__(self, frames=128, S=2, in_ch=3):
        super().__init__()
        self.S = S  # S is the spatial dimension of ST-rPPG block (default 2x2)

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # loop4
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
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

        # ST-rPPG block pooling: adaptively pool to S x S (2 x 2) as in the official paper
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        # x: [Batch, Channel, Length, Height, Width] (NCDHW)
        # Apply per-clip spatio-temporal standardization
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True) + 1e-6
        x = (x - means) / stds

        parity = []
        x = self.start(x)
        x = self.loop1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder1(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder2(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')

        # [B, 1, T, S, S]
        x = self.end(x)

        # Flatten S x S spatial positions
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # [B, 1, T]

        mean_rppg = sum(x_list) / (self.S * self.S)  # [B, 1, T]
        # X: [B, S*S + 1, T]
        X = torch.cat(x_list + [mean_rppg], dim=1)
        rPPG = mean_rppg.squeeze(1)  # [B, T]
        st_block = X[:, :-1, :]  # [B, S*S, T] = [B, 4, T]

        return rPPG, st_block, None, None
