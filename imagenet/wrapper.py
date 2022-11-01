import torch
import torch.nn as nn
import torch.nn.functional as F


class Wrapper(nn.Module):
    def __init__(self, student, teacher):
        super(Wrapper, self).__init__()
        self.t_dims = teacher.fc.in_features
        self.s_dims = student.fc.in_features
        teacher.fc = nn.Identity()
        student.fc = nn.Identity()

        self.backbone = student
        print(self.t_dims, self.s_dims, 'teacher-student hidden dims. and use_proj')
        if self.t_dims != self.s_dims:
            self.proj_head = nn.Sequential(
                nn.Linear(self.s_dims, self.t_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.t_dims, self.t_dims)
            )
        else:
            self.proj_head = None

    def forward(self, x):
        last_feat = self.backbone(x)
        if self.proj_head is not None:
            last_feat = self.proj_head(last_feat)
        return last_feat
