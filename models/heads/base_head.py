# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseHead(nn.Module):
    """Base head."""

    def __init__(self, init_cfg=None):
        super(BaseHead, self).__init__(init_cfg)

    @abstractmethod
    def forward_train(self, x, labels, **kwargs):
        pass
