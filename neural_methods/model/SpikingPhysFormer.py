import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import math

class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):
        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
            else:
                return out_normal

class S3A(nn.Module):
    """
    Simplified Spiking Self-Attention (S3A)
    """
    def __init__(self, dim, num_heads, theta=0.7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query uses Temporal Difference Convolution (TDC)
        self.q_conv = nn.Sequential(
            CDC_T(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim)
        )
        self.q_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        
        # Key uses vanilla 3D convolution
        self.k_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim)
        )
        self.k_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        
        # Value uses normalized spikes directly (handled in forward)
        self.v_bn = nn.BatchNorm3d(dim)
        self.v_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

        self.out_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

    def forward(self, x):
        # x shape: [T_s, B, C, T, H, W]
        T_s, B, C, T, H, W = x.shape
        
        # Process each SNN timestep
        q_list = []
        k_list = []
        v_list = []
        for t in range(T_s):
            q_list.append(self.q_conv(x[t]))
            k_list.append(self.k_conv(x[t]))
            v_list.append(self.v_bn(x[t]))
        
        q = torch.stack(q_list) # [T_s, B, C, T, H, W]
        k = torch.stack(k_list)
        v = torch.stack(v_list)
        
        q = self.q_sn(q)
        k = self.k_sn(k)
        v = self.v_sn(v)
        
        # Reshape for multi-head attention
        # [T_s, B, C, T, H, W] -> [T_s, B, num_heads, head_dim, T*H*W]
        N = T * H * W
        q = q.reshape(T_s, B, self.num_heads, self.head_dim, N)
        k = k.reshape(T_s, B, self.num_heads, self.head_dim, N)
        v = v.reshape(T_s, B, self.num_heads, self.head_dim, N)
        
        # S3A'(Q, K, V) = SN(SUM_c(Q * K)) * V
        # Q * K is Hadamard product
        attn_weight = q * k # [T_s, B, num_heads, head_dim, N]
        attn_weight = attn_weight.sum(dim=-1, keepdim=True) # [T_s, B, num_heads, head_dim, 1]
        attn_weight = self.out_sn(attn_weight)
        
        out = attn_weight * v # [T_s, B, num_heads, head_dim, N]
        out = out.reshape(T_s, B, C, T, H, W)
        
        return out

class SpikingMLP(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim)
        )
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim)
        )
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

    def forward(self, x):
        # x: [T_s, B, C, T, H, W]
        T_s = x.shape[0]
        out_list = []
        for t in range(T_s):
            h = self.fc1(x[t])
            h = self.sn1(h)
            h = self.fc2(h)
            h = self.sn2(h)
            out_list.append(h)
        return torch.stack(out_list)

class ParallelSpikeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, theta=0.7):
        super().__init__()
        self.s3a = S3A(dim, num_heads, theta=theta)
        self.mlp = SpikingMLP(dim, ff_dim)
        
        # Residual connection in SNN usually involves membrane potential
        # The paper says: U_l = S3A(S_{l-1}) + MLP(S_{l-1}) + U_{l-1}
        # S_{l-1} are input spikes, U is membrane potential.
        # spikingjelly neurons handle membrane potential internally.
        
    def forward(self, x):
        # x is spikes from previous layer: [T_s, B, C, T, H, W]
        attn_out = self.s3a(x)
        mlp_out = self.mlp(x)
        
        # Parallel sum + residual
        return x + attn_out + mlp_out

class SpikingPhysFormer(nn.Module):
    def __init__(self, num_layers=4, dim=96, num_heads=4, ff_dim=144, theta=0.7, T_s=4, frame=160):
        super().__init__()
        self.dim = dim
        self.T_s = T_s
        self.frame = frame

        # ANN Stem
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        # Patch Embedding (ft=4, fh=4, fw=4 in SpikingPhysFormer if following 32x downsampling spatially)
        # 128 -> 4 is 32x. Stem0,1,2 each 2x maxpool = 8x. Patch embedding needs 4x.
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(4, 4, 4), stride=(4, 4, 4))
        
        # SNN Input Spike Generator (just a LIF node to start the spikes from ANN features)
        self.input_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

        # Parallel Spike Transformer Blocks
        self.blocks = nn.ModuleList([
            ParallelSpikeTransformerBlock(dim, num_heads, ff_dim, theta=theta)
            for _ in range(num_layers)
        ])

        # Predictor Head
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, T, C, H, W] in the toolbox data loader
        # need [B, C, T, H, W] for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        
        # If data contains both DiffNormalized and Raw (6 channels), 
        # take only the first 3 (DiffNormalized)
        if x.shape[1] == 6:
            x = x[:, :3, :, :, :]
            
        B, C, T, H, W = x.shape
        
        # ANN Stem
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x) # [B, dim, T/4, H/32, W/32]
        
        # Direct Encoding: replicate for T_s timesteps
        # [B, C, T', H', W'] -> [T_s, B, C, T', H', W']
        x = x.unsqueeze(0).repeat(self.T_s, 1, 1, 1, 1, 1)
        
        # Initial spikes
        x = self.input_sn(x)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        # Predict head: mean over T_s
        # x: [T_s, B, C, T', H', W']
        x = x.mean(dim=0) # [B, C, T', H', W']
        
        x = self.upsample(x)
        x = self.upsample2(x) # [B, dim/2, T, H/32, W/32]
        
        x = torch.mean(x, dim=3) # [B, dim/2, T, W/32]
        x = torch.mean(x, dim=3) # [B, dim/2, T]
        
        rPPG = self.ConvBlockLast(x) # [B, 1, T]
        return rPPG.squeeze(1)
