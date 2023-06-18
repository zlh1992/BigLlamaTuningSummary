# -*- coding: utf-8 -*-
# @Time : 2022/11/23 下午4:05
# @Author : zlh1992
# @Email : zlh1992@126.com
# @File : focal_loss.py
# @Project : WSDM_2023

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction,
                                            ignore_index=self.ignore_index)
        return loss
