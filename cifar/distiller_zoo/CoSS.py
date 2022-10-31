import torch
import torch.nn as nn
import torch.nn.functional as F


class CoSSLossPreN(nn.Module):
    def __init__(self, args):
        super(CoSSLossPreN, self).__init__()
        self.t_temp = args.t_temp
        self.s_temp = args.s_temp
        self.w_0 = 1
        self.w_1 = args.w_coss
        self.method = args.distill

    def forward(self, s_feat, t_feat):
        def dot_p(t_feat, s_feat):
            tf = F.normalize(t_feat/self.t_temp, dim=-1, p=2)
            sf = F.normalize(s_feat/self.s_temp, dim=-1, p=2)
            batchloss = -(tf * sf).sum(dim=-1)
            return batchloss

        if self.method == 'cos_pre':
            return torch.mean(dot_p(t_feat, s_feat))
        else:
            return self.w_0 * torch.mean(dot_p(t_feat, s_feat)) + self.w_1 * torch.mean(
                dot_p(t_feat.T, s_feat.T))


class CoSSLossPostN(CoSSLossPreN):

    def forward(self, s_feat, t_feat):
        def dot_p(t_feat, s_feat):
            tf = F.normalize(t_feat, dim=-1, p=2) / self.t_temp
            sf = F.normalize(s_feat, dim=-1, p=2) / self.s_temp
            batchloss = -(tf * sf).sum(dim=-1)
            return batchloss

        if self.method == 'cos_post':
            return torch.mean(dot_p(t_feat, s_feat))
        else:
            return self.w_0 * torch.mean(dot_p(t_feat, s_feat)) + self.w_1 * torch.mean(
                dot_p(t_feat.T, s_feat.T))