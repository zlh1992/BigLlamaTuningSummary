# -*- coding: utf-8 -*-
# @Time : 2022/11/23 下午4:05
# @Author : zlh1992
# @Email : zlh1992@126.com
# @File : rdrop.py
# @Project : WSDM_2023


import torch
from torch import nn
import torch.nn.functional as F


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
