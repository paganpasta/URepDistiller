import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, args):
        super(DINOLoss, self).__init__()
        self.t_temp = args.t_temp
        self.s_temp = args.s_temp

    def forward(self, s_feat, t_feat):
        s_feat = s_feat / self.s_temp
        t_feat = t_feat / self.t_temp

        targets = F.softmax(t_feat, dim=-1)
        logprobs = F.log_softmax(s_feat.view(s_feat.shape[0], -1), dim=1)
        batchloss = - torch.sum(targets.view(targets.shape[0], -1) * logprobs, dim=1)
        return torch.mean(batchloss)
