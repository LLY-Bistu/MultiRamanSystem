# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS


@HEADS.register_module()
class MultiTaskLinearClsHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(MultiTaskLinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # Task 1 (Classify the degree of differentiation of cancer cells)
        self.fc1 = nn.Linear(self.in_channels, 5)
        # Task 2 (Classifying TNM)
        self.fc2 = nn.Sequential(
            nn.Linear(self.in_channels, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 7),
        )
        # Task 3
        self.fc3 = nn.Sequential(
            nn.Linear(self.in_channels, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 5),
        )

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=False):
        x = self.pre_logits(x)
        cls_score1 = self.fc1(x)
        cls_score2 = self.fc2(x)
        cls_score3 = self.fc3(x)
        if softmax:
            pred1 = (F.softmax(cls_score1, dim=1) if cls_score1 is not None else None)
            pred2 = (F.softmax(cls_score2, dim=1) if cls_score2 is not None else None)
            pred3 = (F.softmax(cls_score3, dim=1) if cls_score2 is not None else None)
        else:
            pred1 = cls_score1
            pred2 = cls_score2
            pred3 = cls_score3
        pred = torch.cat((pred1, pred2, pred3), dim=1)
        pred = list(pred.detach().cpu())

        return pred


    def forward_train(self, x, labels, **kwargs):
        x = self.pre_logits(x)
        cls_score1 = self.fc1(x)
        cls_score2 = self.fc2(x)
        cls_score3 = self.fc3(x)

        labels1 = labels[:, 0]
        labels2 = labels[:, 1]
        labels3 = labels[:, 2]
        losses1 = self.loss(cls_score1, labels1.long(), **kwargs)
        losses2 = self.loss(cls_score2, labels2.long(), **kwargs)
        losses3 = self.loss(cls_score3, labels3.long(), **kwargs)

        return losses1, losses2, losses3
