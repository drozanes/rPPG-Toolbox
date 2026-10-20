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
    Official Contrast-Phys+ Spatio-Temporal Contrastive Loss (TPAMI 2024).
    Supports both unsupervised and weakly-supervised learning.
    """
    def __init__(self, config):
        super(ContrastPhysLoss, self).__init__()
        self.fs = config.TRAIN.DATA.FS
        contrast_cfg = getattr(config.MODEL, 'CONTRAST_PHYS', None)
        self.delta_t = getattr(contrast_cfg, 'DELTA_T', 64) if contrast_cfg else 64
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

    def compare_samples(self, list_a, list_b, exclude_same=False):
        """
        Computes pairwise mean squared error between sample sets list_a and list_b.
        Vectorized implementation of Eq. 5 and Eq. 7 of TPAMI 2024 paper.
        """
        # list_a: [N, F], list_b: [M, F]
        N, F_dim = list_a.shape
        M, _ = list_b.shape

        diff = list_a.unsqueeze(1) - list_b.unsqueeze(0)  # [N, M, F]
        sq_diff = torch.mean(diff ** 2, dim=-1)  # [N, M]

        if exclude_same:
            mask = ~torch.eye(N, dtype=torch.bool, device=list_a.device)
            return torch.mean(sq_diff[mask])
        else:
            return torch.mean(sq_diff)

    def forward(self, st_block, labels=None):
        # st_block: [B, N_spatial, T] (e.g. [B, 4, T])
        if st_block.ndim == 2:
            st_block = st_block.unsqueeze(1)

        B, num_spatial, T = st_block.shape

        # Ensure temporal sampling is ALWAYS active
        if self.delta_t >= T:
            actual_delta_t = max(16, T // 2)
        else:
            actual_delta_t = self.delta_t

        # 1. Spatiotemporal sampling on ST-rPPG block
        all_samples = []
        for b in range(B):
            samples_per_video = []
            for c in range(num_spatial):
                for _ in range(self.K):
                    if T > actual_delta_t:
                        offset = torch.randint(0, T - actual_delta_t + 1, (1,), device=st_block.device).item()
                        segment = st_block[b, c, offset : offset + actual_delta_t]
                    else:
                        segment = st_block[b, c, :]
                    samples_per_video.append(self.norm_psd(segment))
            all_samples.append(torch.stack(samples_per_video))  # [N, F_dim], N = num_spatial * K

        # 2. Positive rPPG-rPPG Loss (L_p^{RR})
        # Samples from the same video must share identical heart rate PSD
        pos_loss = sum(self.compare_samples(all_samples[b], all_samples[b], exclude_same=True) for b in range(B)) / B

        # 3. Negative rPPG-rPPG Loss (L_n^{RR})
        # Samples from different videos must have distinct PSDs
        if B >= 2:
            neg_loss = 0.0
            num_pairs = 0
            for i in range(B):
                for j in range(i + 1, B):
                    neg_loss -= self.compare_samples(all_samples[i], all_samples[j], exclude_same=False)
                    num_pairs += 1
            neg_loss = neg_loss / num_pairs
        else:
            neg_loss = torch.tensor(0.0, device=st_block.device)

        total_loss = pos_loss + neg_loss

        # 4. Weak Supervision with Ground Truth (L_p^{GR} and L_n^{GR})
        if labels is not None:
            if labels.ndim > 2:
                labels = labels.squeeze(1)

            gt_samples = []
            for b in range(B):
                gt_sig = labels[b]
                samples_per_gt = []
                for _ in range(num_spatial * self.K):
                    if gt_sig.shape[-1] > actual_delta_t:
                        offset = torch.randint(0, gt_sig.shape[-1] - actual_delta_t + 1, (1,), device=labels.device).item()
                        gt_segment = gt_sig[offset : offset + actual_delta_t]
                    else:
                        gt_segment = gt_sig
                    samples_per_gt.append(self.norm_psd(gt_segment))
                gt_samples.append(torch.stack(samples_per_gt))

            # GT-rPPG Positive Loss
            pos_loss_gt = sum(self.compare_samples(all_samples[b], gt_samples[b], exclude_same=False) for b in range(B)) / B

            # GT-rPPG Negative Loss
            if B >= 2:
                neg_loss_gt = 0.0
                for i in range(B):
                    for j in range(i + 1, B):
                        neg_loss_gt -= 0.5 * (
                            self.compare_samples(all_samples[i], gt_samples[j], exclude_same=False) +
                            self.compare_samples(all_samples[j], gt_samples[i], exclude_same=False)
                        )
                neg_loss_gt = neg_loss_gt / num_pairs
            else:
                neg_loss_gt = torch.tensor(0.0, device=st_block.device)

            total_loss = total_loss + pos_loss_gt + neg_loss_gt

        return total_loss
