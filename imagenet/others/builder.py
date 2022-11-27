import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_p(t_feat, s_feat, t_temp, s_temp):
    tf = F.normalize(t_feat / t_temp, dim=-1, p=2)
    sf = F.normalize(s_feat / s_temp, dim=-1, p=2)
    batchloss = -(tf * sf).sum(dim=-1)
    return batchloss


class OTHERS(nn.Module):
    """
    For methods such as DINO, COS and COSS
    """
    def __init__(self, student, teacher, dim=128, t_temp=0.02, s_temp=0.7, **kwargs):
        """
        dim:        IGNORED
        t_temp:  teacher temperature
        s_temp: student temperature
        """
        super(OTHERS, self).__init__()

        self.t_temp = t_temp
        self.s_temp = s_temp
        self.dim = dim

        # create the Teacher/Student encoders
        # num_classes is the output fc dimension
        student = student(num_classes=dim)
        teacher = teacher(num_classes=dim)

        t_dims = teacher.fc.in_features
        s_dims = student.fc.in_features
        student.fc = nn.Identity()
        teacher.fc = nn.Identity()

        self.student = student
        self.teacher = teacher

        if s_dims != t_dims:
            self.proj = nn.Sequential(
                nn.Linear(s_dims, t_dims),
                nn.ReLU(inplace=True),
                nn.Linear(t_dims, t_dims)
            )
        else:
            self.proj = None

        # not update by gradient
        for param_k in self.teacher.parameters():
            param_k.requires_grad = False


class COSS(OTHERS):
    def forward(self, x):
        s_emb = self.student(x)
        with torch.no_grad():
            t_emb = self.teacher(x)
        f_sim = dot_p(t_emb, s_emb, self.t_temp, self.s_temp)
        s_sim = dot_p(t_emb.T, s_emb.T, self.t_temp, self.s_temp)
        return f_sim.mean() + s_sim.mean()


class COS(OTHERS):
    def forward(self, x):
        s_emb = self.student(x)
        with torch.no_grad():
            t_emb = self.teacher(x)
        f_sim = dot_p(t_emb, s_emb, self.t_temp, self.s_temp)
        return f_sim.mean()


class DINO(OTHERS):

    def forward(self, x):
        s_emb = self.student(x)
        with torch.no_grad():
            t_emb = self.teacher(x)
        s_feat = s_emb / self.s_temp
        t_feat = t_emb / self.t_temp

        targets = F.softmax(t_feat, dim=-1)
        logprobs = F.log_softmax(s_feat.view(s_feat.shape[0], -1), dim=1)
        batchloss = - torch.sum(targets.view(targets.shape[0], -1) * logprobs, dim=1)
        return torch.mean(batchloss)
