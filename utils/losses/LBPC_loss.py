import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# @torch.jit.script
def cosine(input1, input2):
    input1 = F.normalize(input1, dim=1)
    input1 = input1.unsqueeze(1)
    input2 = F.normalize(input2, dim=1)
    input2 = input2.unsqueeze(2)
    similarity = torch.bmm(input1, input2).squeeze().squeeze()
    return similarity


class SpearmanCorrelationLoss(nn.Module):
    def __init__(self, temp=0.2):
        super(SpearmanCorrelationLoss, self).__init__()
        self.temp = temp

    def forward(self, input1, input2, label, return_sim=False):
        similarity = cosine(input1, input2)
        similarity_sm = F.softmax(similarity / self.temp, dim=0)
        similarity_sm = similarity_sm - torch.mean(similarity_sm)
        label_f = torch.as_tensor(label, dtype=torch.float32) - torch.mean(torch.as_tensor(label, dtype=torch.float32))
        t_m1 = torch.sqrt(torch.sum(similarity_sm ** 2))
        t_m2 = torch.sqrt(torch.sum(label_f ** 2))
        correlation = torch.sum(similarity_sm * label_f) / torch.clamp(t_m1 * t_m2, 1e-8, np.inf)
        if return_sim:
            return -correlation, similarity
        else:
            return -correlation