import torch
import torch.nn as nn
import torch.nn.functional as F

class CalculateNormPSD(nn.Module):
    """
    Computes normalized Power Spectral Density (PSD) within physiological heart rate range [40, 250] bpm.
    Reuses Gideon & Stent (ICCV 2021) / Sun & Li (TPAMI 2024 / ECCV 2022).
    """
    def __init__(self, Fs=30, high_pass=40, low_pass=250, zero_pad=2):
        super(CalculateNormPSD, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.zero_pad = zero_pad

    def forward(self, x):
        # x: [..., T]
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if self.zero_pad > 0:
            L = x.shape[-1]
            pad_len = int(self.zero_pad * L / 2)
            x = F.pad(x, (pad_len, pad_len), 'constant', 0)

        # Power spectrum via real FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        psd = torch.abs(x_fft) ** 2

        # Frequency filter [high_pass/60, low_pass/60] Hz
        T_curr = x.shape[-1]
        freqs = torch.fft.rfftfreq(T_curr, d=1.0 / self.Fs).to(x.device)
        mask = (freqs >= (self.high_pass / 60.0)) & (freqs <= (self.low_pass / 60.0))

        # Filter PSD to physiological frequencies
        psd = psd[..., mask]

        # Normalize PSD to sum to 1
        psd_sum = torch.sum(psd, dim=-1, keepdim=True) + 1e-9
        psd_norm = psd / psd_sum
        return psd_norm


class IrrelevantPowerRatio(nn.Module):
    """
    Irrelevant Power Ratio (IPR) from Gideon et al. (ICCV 2021) and Sun & Li (TPAMI 2024).
    Measures the ratio of power outside the physiological HR range [40, 250] bpm to total power.
    Used for model selection to prevent picking collapsed checkpoints.
    """
    def __init__(self, Fs=30, high_pass=40, low_pass=250):
        super(IrrelevantPowerRatio, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds):
        # preds: [B, T]
        x_fft = torch.fft.rfft(preds, dim=-1)
        power = torch.abs(x_fft) ** 2

        T_curr = preds.shape[-1]
        freqs = torch.fft.rfftfreq(T_curr, d=1.0 / self.Fs).to(preds.device)
        use_freqs = (freqs >= (self.high_pass / 60.0)) & (freqs <= (self.low_pass / 60.0))
        zero_freqs = ~use_freqs

        use_energy = torch.sum(power[..., use_freqs], dim=-1)
        zero_energy = torch.sum(power[..., zero_freqs], dim=-1)
        denom = use_energy + zero_energy + 1e-9

        energy_ratio = zero_energy / denom
        return energy_ratio


class ContrastPhysLoss(nn.Module):
    """
    Robust Spatio-Temporal Contrastive Loss for ContrastPhys / Contrast-Phys+ (TPAMI 2024 / ECCV 2022).
    Uses centroid-anchored positive intra-video attraction and inter-video centroid repulsion.
    Supports both unsupervised learning and weakly-supervised learning with contact PPG.
    """
    def __init__(self, config):
        super(ContrastPhysLoss, self).__init__()
        self.fs = config.TRAIN.DATA.FS
        contrast_cfg = getattr(config.MODEL, 'CONTRAST_PHYS', None)
        self.delta_t = getattr(contrast_cfg, 'DELTA_T', 128) if contrast_cfg else 128
        self.K = getattr(contrast_cfg, 'K', 4) if contrast_cfg else 4
        self.high_pass = getattr(contrast_cfg, 'HIGH_PASS', 40.0) if contrast_cfg else 40.0
        self.low_pass = getattr(contrast_cfg, 'LOW_PASS', 250.0) if contrast_cfg else 250.0
        self.zero_pad = getattr(contrast_cfg, 'ZERO_PAD', 2) if contrast_cfg else 2
        
        self.norm_psd = CalculateNormPSD(
            Fs=self.fs, 
            high_pass=self.high_pass, 
            low_pass=self.low_pass, 
            zero_pad=self.zero_pad
        )

    def forward(self, st_block, labels=None):
        # st_block: [B, N_spatial, T]
        if st_block.ndim == 2:
            st_block = st_block.unsqueeze(1)

        B, num_spatial, T = st_block.shape
        actual_delta_t = self.delta_t if self.delta_t <= T else T

        # Vectorized temporal sampling across all spatial regions
        if actual_delta_t < T:
            offsets = [torch.randint(0, T - actual_delta_t + 1, (1,), device=st_block.device).item() for _ in range(self.K)]
            segments = torch.stack([st_block[:, :, o : o + actual_delta_t] for o in offsets], dim=2)  # [B, N, K, delta_t]
            psds = self.norm_psd(segments)  # [B, N, K, F]
        else:
            psds = self.norm_psd(st_block).unsqueeze(2)  # [B, N, 1, F]

        B_c, N_c, K_c, F_c = psds.shape
        psds_flat = psds.view(B_c, N_c * K_c, F_c)
        mean_psd = torch.mean(psds_flat, dim=1, keepdim=True)  # [B, 1, F]

        # 1. Positive rPPG-rPPG Intra-video Loss (L_p^{RR}): Pull each patch PSD to video centroid
        intra_loss = torch.mean((psds_flat - mean_psd) ** 2)

        # 2. Weak Supervision: Positive GT-rPPG Intra-video Loss (L_p^{GR})
        if labels is not None:
            labels_2d = labels.view(B_c, T)
            if actual_delta_t < T:
                gt_offsets = [torch.randint(0, T - actual_delta_t + 1, (1,), device=labels.device).item() for _ in range(self.K)]
                gt_segments = torch.stack([labels_2d[:, o : o + actual_delta_t] for o in gt_offsets], dim=1)  # [B, K, delta_t]
                gt_psd = self.norm_psd(gt_segments)  # [B, K, F]
            else:
                gt_psd = self.norm_psd(labels_2d).unsqueeze(1)  # [B, 1, F]
            mean_gt_psd = torch.mean(gt_psd, dim=1, keepdim=True)  # [B, 1, F]
            intra_loss = intra_loss + torch.mean((psds_flat - mean_gt_psd) ** 2)

        # 3. Negative Inter-video Repulsion Loss (L_n^{RR} and L_n^{GR})
        if B_c > 1:
            v_means = mean_psd.squeeze(1)  # [B, F]
            diff = v_means.unsqueeze(1) - v_means.unsqueeze(0)
            sq_diff = torch.mean(diff ** 2, dim=-1)
            mask = ~torch.eye(B_c, dtype=torch.bool, device=st_block.device)
            neg_loss = - torch.mean(sq_diff[mask])

            if labels is not None:
                v_gt_means = mean_gt_psd.squeeze(1)
                diff_gr = v_means.unsqueeze(1) - v_gt_means.unsqueeze(0)
                sq_diff_gr = torch.mean(diff_gr ** 2, dim=-1)
                neg_loss_gr = - torch.mean(sq_diff_gr[mask])
                inter_loss = (neg_loss + 2.0 * neg_loss_gr) / 3.0
            else:
                inter_loss = neg_loss
        else:
            inter_loss = torch.tensor(0.0, device=st_block.device)

        total_loss = intra_loss + inter_loss
        return total_loss
