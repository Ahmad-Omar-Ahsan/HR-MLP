# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from models.manifolds.lorentz import Lorentz
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph, LorentzDenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import math


def DropPath(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class LorentzLinear(nn.Module):
    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        bias=True,
        dropout=0.1,
        scale=10,
        fixscale=False,
        nonlin=None,
    ):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(
            torch.ones(()) * math.log(scale), requires_grad=not fixscale
        )

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / (x_narrow * x_narrow).sum(
            dim=-1, keepdim=True
        ).clamp_min(
            1e-8
        )  # Scale function
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class FFN_Lorentz(nn.Module):
    def __init__(
        self, manifold, in_features, out_features, use_bias, dropout, nonlin=None
    ):
        super(FFN_Lorentz, self).__init__()
        self.linear_1 = LorentzLinear(
            manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin
        )
        self.linear_2 = LorentzLinear(
            manifold, in_features // 2, out_features, use_bias, dropout, nonlin=nonlin
        )
        self.batchnorm = nn.BatchNorm2d(in_features)
        self.rearrange1 = Rearrange(
            "b c h w -> b (h w) c",
        )
        self.l = Lorentz()
        self.to_euclid = self.l.logmap0
        self.to_lorentz = self.l.expmap0

    def forward(self, x):
        # B, C, H, W = x.shape
        # x = self.rearrange1(x)
        x = self.linear_1(x)
        # x = self.to_lorentz(self.batchnorm(self.to_euclid(x)))
        x = self.linear_2(x)
        # x = self.to_lorentz(self.batchnorm(self.to_euclid(x)))
        # x = x.reshape(B,C,H,W)
        return x


class LMRConv2d(nn.Module):
    """
    Lorentz Max-Relative Graph Convolution
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(LMRConv2d, self).__init__()
        self.nn = FFN_Lorentz(
            manifold="Lorentz",
            in_features=2 * in_channels,
            out_features=out_channels,
            nonlin=act_layer(act),
            use_bias=True,
            dropout=0,
        )
        self.l = Lorentz()

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        x = rearrange(x, "b c n x -> b (x n) c")
        x = self.nn(x)
        x = x.reshape(b, c, n, _)
        return x


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(
            self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True
        )
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(
        self, in_channels, out_channels, conv="edge", act="relu", norm=None, bias=True
    ):
        super(GraphConv2d, self).__init__()
        if conv == "edge":
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "mr":
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "lmr":
            self.gconv = LMRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "sage":
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == "gin":
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError("conv:{} is not supported".format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class LorentzDyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        dilation=1,
        conv="",
        act="relu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        r=1,
    ):
        super(LorentzDyGraphConv2d, self).__init__(
            in_channels, out_channels, conv, act, norm, bias
        )
        self.k = kernel_size
        self.d = dilation
        self.r = r

        self.lorentz_dilated_knn_graph = LorentzDenseDilatedKnnGraph(
            kernel_size, dilation, stochastic, epsilon
        )

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None

        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.lorentz_dilated_knn_graph(x, y, relative_pos)
        x = super(LorentzDyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        dilation=1,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        r=1,
    ):
        super(DyGraphConv2d, self).__init__(
            in_channels, out_channels, conv, act, norm, bias
        )
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(
            kernel_size, dilation, stochastic, epsilon
        )

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(
        self,
        in_channels,
        kernel_size=9,
        dilation=1,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        r=1,
        n=196,
        drop_path=0.0,
        relative_pos=False,
    ):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(
            in_channels,
            in_channels * 2,
            kernel_size,
            dilation,
            conv,
            act,
            norm,
            bias,
            stochastic,
            epsilon,
            r,
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            # print("using relative_pos")
            relative_pos_tensor = (
                torch.from_numpy(
                    np.float32(get_2d_relative_pos_embed(in_channels, int(n**0.5)))
                )
                .unsqueeze(0)
                .unsqueeze(1)
            )
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor,
                size=(n, n // (r * r)),
                mode="bicubic",
                align_corners=False,
            )
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False
            )

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(
                relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic"
            ).squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
