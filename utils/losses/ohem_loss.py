# -*- coding: utf-8 -*-
# @Time : 2022/11/23 下午4:05
# @Author : zlh1992
# @Email : zlh1992@126.com
# @File : ohem_loss.py
# @Project : WSDM_2023
import torch
import torch.nn as nn


class BCELogitsWithOHEM(nn.Module):

    def __init__(self, neg_ratio: int = 3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduce=False)
        self.neg_ratio = neg_ratio
        self.eps = 1e-7

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        pos_cnt = int(labels.sum())
        neg_cnt = int(min((1 - labels).sum(), pos_cnt * self.neg_ratio))

        bce_loss = self.bce(logits, labels.float())
        pos_loss = bce_loss * labels
        neg_loss = bce_loss * (1 - labels)
        neg_loss, _ = torch.topk(neg_loss.view(-1), neg_cnt)

        balance_loss = (pos_loss.sum() + neg_loss.sum()) / (pos_cnt + neg_cnt + self.eps)
        return balance_loss
