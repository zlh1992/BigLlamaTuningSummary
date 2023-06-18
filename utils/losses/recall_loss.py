# -*- coding: utf-8 -*-
# @Time : 2022/11/23 下午4:05
# @Author : zlh1992
# @Email : zlh1992@126.com
# @File : recall_loss.py
# @Project : WSDM_2023

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """

    def __init__(self, weight=None):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

    def forward(self, input, target):
        N, C = input.size()[:2]
        logpt = F.log_softmax(input, dim=1)
        pt = logpt.exp()

        ## convert target (N, 1, *) into one hot vector (N, C, *)
        target = target.view(N, 1, -1)  # (N, 1, *)
        target_onehot = torch.zeros(pt.size()).type_as(pt)  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        true_positive = torch.sum(pt * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input)
                recall = recall * self.weight * C  # (N, C)
        recall_loss = 1 - torch.mean(recall)  # 1

        return recall_loss

# if __name__ == '__main__':
#     y_target = torch.Tensor([[0, 1], [1, 0]]).long().cuda(0)
#     y_predict = torch.Tensor([[[1.5, 1.0], [2.8, 1.6]],
#                            [[1.0, 1.0], [2.4, 0.3]]]
#                           ).cuda(0)

#     criterion = RecallLoss(weight=[1, 1])
#     loss = criterion(y_predict, y_target)
#     print(loss)
