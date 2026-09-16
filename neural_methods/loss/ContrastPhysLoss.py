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
        x = x - torch.mean(x, dim=-1, keepdim=True)
        psd = torch.abs(torch.fft.rfft(x, n=T))**2
        freq = torch.fft.rfftfreq(T, d=1/self.fs)
        mask = (freq >= self.high_pass / 60) & (freq <= self.low_pass / 60)
        psd = psd * mask.to(psd.device)
        psd_sum = torch.sum(psd, dim=-1, keepdim=True) + 1e-9
        psd_norm = psd / psd_sum
        return psd_norm

class ContrastPhysLoss(nn.Module):
    def __init__(self, config):
        super(ContrastPhysLoss, self).__init__()
        self.fs = config.TRAIN.DATA.FS
        self.norm_psd = CalculateNormPSD(fs=self.fs)
        self.delta_t = config.MODEL.CONTRAST_PHYS.DELTA_T
        self.K = config.MODEL.CONTRAST_PHYS.K
        
    def forward(self, st_block, labels=None):
        B, N, T = st_block.shape
        total_loss = 0
        
        video_mean_psds = []
        video_mean_gt_psds = []
        
        intra_loss = 0
        
        for b in range(B):
            # 1. Temporal sampling for rPPG signals
            psds = []
            for n in range(N):
                for k in range(self.K):
                    if T > self.delta_t:
                        offset = torch.randint(0, T - self.delta_t + 1, (1,))
                        segment = st_block[b, n, offset : offset + self.delta_t]
                    else:
                        segment = st_block[b, n, :]
                    psds.append(self.norm_psd(segment))
            psds = torch.stack(psds)
            mean_psd = torch.mean(psds, dim=0)
            video_mean_psds.append(mean_psd)
            
            # L_p^{RR}: Intra-video rPPG-rPPG Positive Loss
            intra_loss += torch.mean((psds - mean_psd)**2)
            
            if labels is not None:
                # 2. Temporal sampling for GT signals (Matching the paper!)
                gt_psds = []
                # Sample the same number of temporal chunks as rPPG
                for _ in range(N * self.K):
                    if T > self.delta_t:
                        offset = torch.randint(0, T - self.delta_t + 1, (1,))
                        gt_segment = labels[b, offset : offset + self.delta_t]
                    else:
                        gt_segment = labels[b, :]
                    gt_psds.append(self.norm_psd(gt_segment))
                gt_psds = torch.stack(gt_psds)
                mean_gt_psd = torch.mean(gt_psds, dim=0)
                video_mean_gt_psds.append(mean_gt_psd)
                
                # L_p^{GR}: Intra-video GT-rPPG Positive Loss
                intra_loss += torch.mean((psds - mean_gt_psd)**2)

        intra_loss /= B
        
        # Inter-video losses
        inter_loss = 0
        if B > 1:
            video_mean_psds = torch.stack(video_mean_psds)
            if labels is not None:
                video_mean_gt_psds = torch.stack(video_mean_gt_psds)
                
            num_pairs = B * (B - 1) / 2
            for i in range(B):
                for j in range(i + 1, B):
                    # L_n^{RR}: Inter-video rPPG-rPPG Negative Loss
                    inter_loss -= torch.mean((video_mean_psds[i] - video_mean_psds[j])**2)
                    
                    # L_n^{GR}: Inter-video GT-rPPG Negative Loss (Added to match the paper!)
                    if labels is not None:
                        inter_loss -= torch.mean((video_mean_psds[i] - video_mean_gt_psds[j])**2)
                        inter_loss -= torch.mean((video_mean_psds[j] - video_mean_gt_psds[i])**2)
            
            # Normalize negative losses
            if labels is not None:
                inter_loss /= (num_pairs * 3) # RR + GR(i,j) + GR(j,i)
            else:
                inter_loss /= num_pairs

        total_loss = intra_loss + inter_loss
        return total_loss
