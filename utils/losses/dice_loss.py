# -*- coding: utf-8 -*-
# @Time : 2022/11/23 下午4:05
# @Author : zlh1992
# @Email : zlh1992@126.com
# @File : dice_loss.py
# @Project : WSDM_2023

import torch
from torch import nn


class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data

    Example:
        >>> criterion = DiceLoss()
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        '''
        input: [N, C]
        target: [N, ]
        '''
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss


class DiceLossWithBCELogits(nn.Module):
    """DiceLossWithBCELogits
    Useful in dealing with unbalanced data

    Example:
        >>> criterion = DiceLossWithBCELogits(smooth=True)

    """

    def __init__(self, smooth=True):
        super().__init__()
        self.smooth = smooth
        self.eps = 1e-7

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        p = torch.sigmoid(logits)

        inter = (p * labels).sum()
        union = (p ** 2).sum() + (labels ** 2).sum() + self.eps
        lambda_ = 1 if (self.smooth and torch.max(labels) == 0) else 0

        dice = 1 - 2.0 * (inter + lambda_) / (union + lambda_)
        return dice
