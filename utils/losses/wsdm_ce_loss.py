import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class WsdmCrossEntropyLoss(nn.Module):
    def __init__(self, temp=50):
        super(WsdmCrossEntropyLoss, self).__init__()
        self.temp = 50
    '''
    anchor: bs * dim 
        eg: 4*768
    positive: sum(positive_lens) * dim
    `   eg: 16*768
    negative: sum(negative_lens) * dim
        eg: 27*768
    positive_lens: bs * 1
        eg: [1,6,4,5]
    negative_lens: bs * 1
        eg: [9,4,9,5]
    '''
    def forward(self, anchor, positive, negative, positive_lens, negative_lens):
        anchors = []
        for i in range(len(positive_lens)):
            anchors.append(anchor[i, :].repeat(positive_lens[i], 1))
        for i in range(len(negative_lens)):
            anchors.append(anchor[i, :].repeat(negative_lens[i], 1))
        anchors = torch.cat(anchors)
        preds = torch.cosine_similarity(anchors, torch.cat((positive, negative)))*self.temp
        loss = nn.CrossEntropyLoss()
        m = nn.Softmax()
        l = loss(m(preds), torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))]).float().to(anchors.device)) / (len(positive) + len(negative))
        return l