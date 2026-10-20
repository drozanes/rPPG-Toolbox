'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/blob/main/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    # Reshape the labels tensor to match the shape of inputs
    labels = labels.view(1, -1)
    
    # Compute the KL Div Loss on log_softmax as in the official PhysFormer repo
    criterion = nn.KLDivLoss(reduction='sum')
    loss = criterion(F.log_softmax(inputs, dim=-1), labels)
    return loss
 
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        device = output.device
        two_pi_n_over_N = torch.autograd.Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True).to(device) / N
        hanning = torch.autograd.Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1).to(device)

        k = k.type(torch.FloatTensor).to(device)
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                            + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4] Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        sum_ca = complex_absolute.sum()
        if sum_ca == 0 or torch.isnan(sum_ca):
            return complex_absolute
        return (1.0 / (sum_ca + 1e-7)) * complex_absolute	# Analogous Softmax operator      
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        device = inputs.device
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        device = inputs.device
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        device = inputs.device
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).to(device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std):
        device = inputs.device
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device)
        target_clamped = torch.clamp(target, min=40.0, max=179.0)
        target_val = float(target_clamped.item())
        
        target_distribution = [normal_sampling(target_val, i, std) for i in range(40, 180)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.tensor(target_distribution, dtype=torch.float, device=device)
        target_distribution = target_distribution / target_distribution.sum()
        
        inputs = inputs.view(1, -1)
        target_idx = (target_clamped - bpm_range[0]).long().view(1)

        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        loss_distribution_kl = kl_loss(ca, target_distribution)
        loss_ce = F.cross_entropy(ca, target_idx)
        
        whole_max_val, whole_max_idx = ca.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        return loss_distribution_kl, loss_ce, torch.abs(target_clamped.view(1) - bpm_range[0] - whole_max_idx)
        