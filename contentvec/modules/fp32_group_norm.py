# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Layer norm done in fp32 (for fp16 training)
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch
from fairseq.pdb import set_trace


class GroupNormMasked(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x, mask=None):
        B, C, L = x.size()
        assert C % self.num_groups == 0
        
        x = x.view(B, self.num_groups, C//self.num_groups, L)
        if mask is None:
            mask = torch.ones_like(x)
        else:
            mask = mask.view(B, 1, 1, L)
        x = x * mask
        lengths = mask.sum(dim=3, keepdim=True)
        
        assert x.size(2)==1
        mean_ = x.mean(dim=3, keepdim=True)
        mean = mean_ * L / lengths

        #var = (((x - mean)**2)*mask).sum(dim=3, keepdim=True) / lengths
        #var = (x**2).sum(dim=3, keepdim=True) / lengths - mean**2
        var = (x.var(dim=3, unbiased=False, keepdim=True) + mean_**2) * L / lengths - mean**2
        var = var.add_(self.eps)

        x = x.add_(-mean.detach())
        x = x.div_(var.sqrt().detach())
        
        x = x.view(B, C, L)
        
        x = x.mul_(self.weight.view(1,-1,1))
        x = x.add_(self.bias.view(1,-1,1))
        
        return x
    
    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)