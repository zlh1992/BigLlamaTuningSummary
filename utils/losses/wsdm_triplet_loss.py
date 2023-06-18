import torch
import torch.nn as nn
from torch.nn import functional as F


class WsdmTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, swap: bool = False, reduction: str = 'mean'):
        super(WsdmTripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction

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
        positive_lens = torch.as_tensor(positive_lens).to(anchor.device)
        negative_lens = torch.as_tensor(negative_lens).to(anchor.device)

        anchors = []
        # for i in range(len(positive_lens)):
        for i, z in enumerate(torch.unbind(positive_lens)):
            anchors.append(anchor[i, :].repeat(negative_lens[i] * z, 1))
        anchors = torch.cat(anchors)

        positives = []
        idx = 0
        # for i in range(len(positive_lens)):
        for i, z in enumerate(torch.unbind(positive_lens)):
            positives.append(positive[idx: idx + z, :].repeat_interleave(negative_lens[i], 0))
            idx = idx + z
        positives = torch.cat(positives)

        negatives = []
        idx = 0
        # for i in range(len(negative_lens)):
        for i, z in enumerate(torch.unbind(negative_lens)):
            negatives.append(negative[idx: idx + z, :].repeat(positive_lens[i], 1))
            idx = idx + z
        negatives = torch.cat(negatives)
        
        triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), 
            margin=self.margin, 
            swap=self.swap, 
            reduction=self.reduction)
        
        loss = triplet_loss(anchors, positives, negatives)

        return loss
