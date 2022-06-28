import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn import Linear
from fairseq.pdb import set_trace

class CondLayerNorm(Module):

    def __init__(self, dim_last, eps=1e-5, dim_spk=256, elementwise_affine=True):
        super(CondLayerNorm, self).__init__()
        self.dim_last = dim_last
        self.eps = eps
        self.dim_spk = dim_spk
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight_ln = Linear(self.dim_spk, 
                                    self.dim_last, 
                                    bias=False)
            self.bias_ln = Linear(self.dim_spk, 
                                  self.dim_last, 
                                  bias=False)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight_ln.weight)
            init.zeros_(self.bias_ln.weight)

    def forward(self, input, spk_emb):
        weight = self.weight_ln(spk_emb)
        bias = self.bias_ln(spk_emb)
        return F.layer_norm(
            input, input.size()[1:], weight, bias, self.eps)

    def extra_repr(self):
        return '{dim_last}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)