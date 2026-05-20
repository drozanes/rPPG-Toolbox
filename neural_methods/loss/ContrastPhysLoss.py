import torch
import torch.nn as nn
import torch.nn.functional as F

class CalculateNormPSD(nn.Module):
    def __init__(self, fs=30, high_pass=40, low_pass=250):
        super(CalculateNormPSD, self).__init__()
        self.fs = fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x):
        # x: [T]
        T = x.shape[-1]
        # Detrend
        x = x - torch.mean(x, dim=-1, keepdim=True)
        
        # FFT
        psd = torch.abs(torch.fft.rfft(x, n=T))**2
        
        # Frequency bins
        freq = torch.fft.rfftfreq(T, d=1/self.fs)
        
        # Mask for HR range [40, 250] BPM -> [0.66, 4.16] Hz
        mask = (freq >= self.high_pass / 60) & (freq <= self.low_pass / 60)
        
        # Apply mask
        psd = psd * mask.to(psd.device)
        
        # Normalize
        psd_sum = torch.sum(psd, dim=-1, keepdim=True) + 1e-9
        psd_norm = psd / psd_sum
        
        return psd_norm

class ContrastPhysLoss(nn.Module):
    def __init__(self, config):
        super(ContrastPhysLoss, self).__init__()
        self.fs = config.TRAIN.DATA.FS
        self.norm_psd = CalculateNormPSD(fs=self.fs)
        self.delta_t = config.MODEL.CONTRAST_PHYS.DELTA_T
        self.K = config.MODEL.CONTRAST_PHYS.K # Number of temporal samples per spatial region
        
    def forward(self, st_block, labels=None):
        # st_block: [B, N, T]
        # labels: [B, T] (optional for weakly supervised)
        B, N, T = st_block.shape
        
        total_loss = 0
        
        # Intra-video loss (Positive pairs)
        intra_loss = 0
        for b in range(B):
            psds = []
            for n in range(N):
                for k in range(self.K):
                    # Randomly sample temporal segment
                    if T > self.delta_t:
                        offset = torch.randint(0, T - self.delta_t + 1, (1,))
                        segment = st_block[b, n, offset : offset + self.delta_t]
                    else:
                        segment = st_block[b, n, :]
                    
                    psd = self.norm_psd(segment)
                    psds.append(psd)
            
            psds = torch.stack(psds) # [N*K, freq_bins]
            # Mean PSD for this video
            mean_psd = torch.mean(psds, dim=0)
            # MSE between each sample PSD and mean PSD
            intra_loss += torch.mean((psds - mean_psd)**2)
            
            # If labels are provided (Weakly Supervised)
            if labels is not None:
                gt_segment = labels[b]
                if T > self.delta_t:
                    # Sample same segments as model? Or just full segment?
                    # Paper says GT-rPPG Positive Loss
                    gt_psd = self.norm_psd(gt_segment)
                    intra_loss += torch.mean((psds - gt_psd)**2)

        intra_loss /= B
        
        # Inter-video loss (Negative pairs - for unsupervised)
        # Pushing PSDs of different videos apart
        inter_loss = 0
        if B > 1:
            video_mean_psds = []
            for b in range(B):
                # We can use the mean_psd from above if we store it
                # For simplicity, let's recompute or refactor
                psds = []
                for n in range(N):
                     # Just sample one segment per spatial region for efficiency in negative loss
                    offset = torch.randint(0, max(1, T - self.delta_t + 1), (1,))
                    segment = st_block[b, n, offset : offset + self.delta_t]
                    psds.append(self.norm_psd(segment))
                video_mean_psds.append(torch.mean(torch.stack(psds), dim=0))
            
            video_mean_psds = torch.stack(video_mean_psds) # [B, freq_bins]
            for i in range(B):
                for j in range(i + 1, B):
                    # Negative MSE (maximize distance)
                    # Or 1/MSE, or exp(-MSE)
                    # Paper uses negative MSE: -MSE(psd_i, psd_j)
                    # We want to minimize -MSE, which is maximize MSE
                    inter_loss -= torch.mean((video_mean_psds[i] - video_mean_psds[j])**2)
            
            inter_loss /= (B * (B - 1) / 2)

        total_loss = intra_loss + inter_loss
        return total_loss
