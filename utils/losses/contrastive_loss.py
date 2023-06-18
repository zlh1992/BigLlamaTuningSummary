import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SimLoss(nn.Module):
    def __init__(self, std=0.3):
        super(SimLoss, self).__init__()
        self.std = std
        self.sim_loss = nn.CrossEntropyLoss()
        
    def forward(self, a, b):
        ra = torch.rand(a.size()).to(a.device, non_blocking=True)
        rb = torch.rand(b.size()).to(b.device, non_blocking=True)
        ra = (ra - 0.5) * self.std * torch.std(a)
        rb = (rb - 0.5) * self.std * torch.std(b)
        a_ = a + ra
        b_ = b + rb
        s_encode = torch.concat([a_, b_], dim=0)
        c_encode = torch.concat([a, b], dim=0)
        logit = torch.matmul(s_encode, c_encode.transpose(-2, -1))  # bs内与其他的远 bs*bs
        target = torch.from_numpy(np.array([i for i in range(s_encode.size(0))])).long().to(s_encode.device, non_blocking=True)  # 交叉熵
        loss_val = self.sim_loss(logit, target) # .mean()
        return loss_val