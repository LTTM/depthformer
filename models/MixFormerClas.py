import torch
from torch import nn

from models.mix_transformer_penc import mit_b0

class MixFormerClas(nn.Module):
    def __init__(self, numclasses=1000, use_mlp=True, penc_after_attn=False) -> None:
        super().__init__()

        self.enc = mit_b0(use_mlp=use_mlp, penc_after_attn=penc_after_attn)
        self.cast = nn.Linear(256, numclasses)

        self.relu = nn.GELU()
        self.squash = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        fs = self.enc(x)
        f = self.squash(fs[-1]).flatten(-3,-1)
        f = self.relu(f)
        c = self.cast(f)
        return c

    def get_params(self, lr, wd):
        decay = []
        no_decay = []
        for n, p in self.named_parameters():
            if 'bias' in n:
                no_decay.append(p)
            elif 'norm' in n:
                no_decay.append(p)
            elif 'penc' in n:
                no_decay.append(p)
            else:
                decay.append(p)
        
        return [{'params': decay, 'lr': lr, 'weight_decay': wd}, {'params': no_decay, 'lr': lr, 'weight_decay': 0}]
