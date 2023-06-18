import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CoSENTLoss(nn.Module):
    def __init__(self, temp_=20):
        super(CoSENTLoss, self).__init__()
        self.temp_ = temp_
        
    def forward(self, y_pred, y_true):
        y_pred = y_pred * self.temp_
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_true = y_true[:, None] < y_true[None, :]
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        y_pred = torch.cat((torch.zeros(1).to(y_pred.device, non_blocking=True), y_pred), dim=0)
        loss = torch.logsumexp(y_pred, dim=0)
        return loss
