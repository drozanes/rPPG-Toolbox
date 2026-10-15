import torch
import torch.nn as nn
import torch.nn.functional as F

class CalculateNormPSD(nn.Module):
    def __init__(self, Fs=30, high_pass=40, low_pass=250):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=3):
        # x shape: [..., T]
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), 'constant', 0)

        # Get PSD using forward normalized rfft
        fft_res = torch.fft.rfft(x, dim=-1, norm='forward')
        psd = torch.abs(fft_res) ** 2

        # Filter PSD strictly within physiological range [40 bpm, 250 bpm]
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, psd.shape[-1], device=x.device)
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        psd_filtered = psd[..., use_freqs]

        # Normalize PSD so sum equals 1
        psd_sum = torch.sum(psd_filtered, dim=-1, keepdim=True) + 1e-9
        psd_norm = psd_filtered / psd_sum
        return psd_norm


class ContrastPhysLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fs = config.TRAIN.DATA.FS
        self.delta_t = getattr(config.MODEL.CONTRAST_PHYS, "DELTA_T", 128)
        self.K = getattr(config.MODEL.CONTRAST_PHYS, "K", 4)
        self.norm_psd = CalculateNormPSD(Fs=self.fs, high_pass=40, low_pass=250)
        self.distance_func = nn.MSELoss(reduction='mean')

    def compare_samples(self, list_a, list_b, exclude_same=False):
        total_distance = 0.0
        M = 0
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if not (exclude_same and i == j):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1
        return total_distance / max(M, 1)

    def forward(self, st_block, labels=None):
        # st_block shape: [B, N, T]
        B, N, T = st_block.shape
        # Use delta_t if T >= delta_t, otherwise sample over T/2 or T
        sample_len = self.delta_t if (self.delta_t <= T) else max(int(T // 2), 1)

        # 1. Spatio-temporal sampling of rPPG
        samples_rppg = []
        for b in range(B):
            samples_per_video = []
            for n in range(N):
                for k in range(self.K):
                    if T > sample_len:
                        offset = torch.randint(0, T - sample_len + 1, (1,), device=st_block.device).item()
                        seg = st_block[b, n, offset : offset + sample_len]
                    else:
                        seg = st_block[b, n, :]
                    samples_per_video.append(self.norm_psd(seg))
            samples_rppg.append(samples_per_video)

        # 2. Intra-video positive loss (L_p^{RR})
        pos_loss = 0.0
        for b in range(B):
            pos_loss += self.compare_samples(samples_rppg[b], samples_rppg[b], exclude_same=True)
        pos_loss /= max(B, 1)

        # 3. Inter-video negative loss (L_n^{RR})
        neg_loss = 0.0
        num_pairs = B * (B - 1) / 2
        if B > 1:
            for i in range(B):
                for j in range(i + 1, B):
                    neg_loss -= self.compare_samples(samples_rppg[i], samples_rppg[j])
            neg_loss /= num_pairs

        # 4. Weak supervision with GT signals if available
        pos_loss_gt = 0.0
        neg_loss_gt = 0.0
        if labels is not None and labels.numel() > 0:
            samples_gt = []
            for b in range(B):
                samples_per_gt = []
                for _ in range(N * self.K):
                    if labels.shape[-1] > sample_len:
                        offset = torch.randint(0, labels.shape[-1] - sample_len + 1, (1,), device=labels.device).item()
                        gt_seg = labels[b, offset : offset + sample_len]
                    else:
                        gt_seg = labels[b, :]
                    samples_per_gt.append(self.norm_psd(gt_seg))
                samples_gt.append(samples_per_gt)

            for b in range(B):
                pos_loss_gt += self.compare_samples(samples_rppg[b], samples_gt[b])
            pos_loss_gt /= max(B, 1)

            if B > 1:
                for i in range(B):
                    for j in range(i + 1, B):
                        neg_loss_gt -= self.compare_samples(samples_rppg[i], samples_gt[j])
                        neg_loss_gt -= self.compare_samples(samples_rppg[j], samples_gt[i])
                neg_loss_gt /= (num_pairs * 2)

        total_loss = pos_loss + neg_loss + pos_loss_gt + neg_loss_gt
        return total_loss
