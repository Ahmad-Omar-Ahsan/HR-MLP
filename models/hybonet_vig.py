import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import Sequential as Seq
from models.gcn_lib.torch_vertex import  act_layer
from models.layers.hyp_layers import LorentzGraphConvolution, HyperbolicGraphConvolution


class Stem(nn.Module):
    """Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, in_dim=3, out_dim=768, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )
        self.rearrange1 = Rearrange(
            "b c h w -> b (h w) c",
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.rearrange1(x)
        return x


class Hybovig_block(nn.Module):
    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, local_agg, nonlin=None):
        pass
        

