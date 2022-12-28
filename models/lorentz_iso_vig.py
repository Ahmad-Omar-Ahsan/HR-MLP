import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import Sequential as Seq

from geoopt import ManifoldParameter
from models.gcn_lib.torch_vertex import DyGraphConv2d, LorentzDyGraphConv2d, act_layer
from models.manifolds.lorentz import Lorentz
import models.manifolds as manifolds



class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x
    

def DropPath(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output




class FFN_Lorentz(nn.Module):
    def __init__(
        self, manifold, in_features, out_features, use_bias, dropout, nonlin=None
    ):
        super(FFN_Lorentz, self).__init__()
        self.linear_1 = LorentzLinear(
            manifold, in_features, in_features, use_bias, dropout, nonlin=nonlin
        )
        self.linear_2 = LorentzLinear(
            manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin
        )
        self.batchnorm = nn.BatchNorm1d(in_features)
        self.rearrange1 = Rearrange(
            "b c h w -> b (h w) c",
        )
        self.l = Lorentz()
        self.to_euclid = self.l.logmap0
        self.to_lorentz = self.l.expmap0

    def forward(self, x):
        temp = x
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x + temp


class Lorentz_grapher(nn.Module):
    """Grapher module with graph conv and FC layers"""

    def __init__(
        self, in_channels, hidden_channels, conv, act, k=9, dilation=1, drop_path=0.0
    ):
        super(Lorentz_grapher, self).__init__()
        self.fc1 = FFN_Lorentz(
            manifold="Lorentz",
            in_features=in_channels,
            out_features=hidden_channels,
            nonlin=act_layer(act),
            use_bias=True,
            dropout=0.0,
        )

        self.graph_conv = LorentzDyGraphConv2d(
            in_channels, hidden_channels, k, dilation, conv=conv, act=act
        )

        self.fc2 = FFN_Lorentz(
            manifold="Lorentz",
            in_features=in_channels,
            out_features=hidden_channels,
            nonlin=act_layer(act),
            use_bias=True,
            dropout=0.0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.rearrange1 = Rearrange(
            "b c h w -> b (h w) c",
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        b, c, h, w = x.shape
        shortcut = x

        x = self.rearrange1(x)
        x = self.fc1(x)
        x = x.reshape(b, c, h, w)
        x = self.graph_conv(x)
        x = self.rearrange1(x)
        x = self.fc2(x)
        x = x.reshape(b, c, h, w)
        x = self.drop_path(x) + shortcut
        return x.reshape(B, C, H, W)


class Grapher(nn.Module):
    """Grapher module with graph conv and FC layers"""

    def __init__(
        self, in_channels, hidden_channels, conv, act, k=9, dilation=1, drop_path=0.0
    ):
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = nn.Sequential(
            DyGraphConv2d(
                in_channels, hidden_channels, k, dilation, conv=conv, act=act
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        # print(x.shape)
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x.reshape(B, C, H, W)


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="relu",
        drop_path=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = drop_path(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class VIG_block_Lorentz_complete(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilation: int,
        conv: str,
        act: str,
        drop_path: float = 0.0,
        drop_out: float = 0.0,
    ):
        """VIG block with Grapher and FFN modules

        Args:
            channels (int): Number of channels
            k (int): Number of neighbors
            dilation (int): Dilation size
            conv (str) : Type of convolution_
            drop_path (float, optional): Drop rate. Defaults to 0.0.
        """
        super(VIG_block_Lorentz_complete, self).__init__()

        self.lorentz_grapher = Lorentz_grapher(
            channels,
            channels,
            k=k,
            dilation=dilation,
            conv=conv,
            act=act,
            drop_path=drop_path,
        )
        self.ffn = FFN_Lorentz(
            manifold="lorentz",
            in_features=channels,
            out_features=channels,
            use_bias=True,
            dropout=drop_out,
            nonlin=act_layer(act),
        )
        l = Lorentz()
        self.to_euclid = l.logmap0
        self.to_lorentz = l.expmap0
        self.rearrange1 = Rearrange(
            "b c h w -> b (h w) c",
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.lorentz_grapher(x)
        x = self.rearrange1(x)
        x = self.ffn(x)
        x = x.reshape(B, C, H, W)
        # print(x.shape)
        return x


class VIG_block_Lorentz(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilation: int,
        conv: str,
        act: str,
        drop_path: float = 0.0,
        drop_out: float = 0.0,
    ):
        """VIG block with Grapher and FFN modules

        Args:
            channels (int): Number of channels
            k (int): Number of neighbors
            dilation (int): Dilation size
            conv (str) : Type of convolution_
            drop_path (float, optional): Drop rate. Defaults to 0.0.
        """
        super(VIG_block_Lorentz, self).__init__()
        self.grapher = Grapher(
            channels,
            channels,
            k=k,
            dilation=dilation,
            conv=conv,
            act=act,
            drop_path=drop_path,
        )
        self.ffn = FFN_Lorentz(
            manifold="lorentz",
            in_features=channels,
            out_features=channels,
            use_bias=True,
            dropout=drop_out,
        )
        l = Lorentz()
        self.to_euclid = l.logmap0
        self.to_lorentz = l.expmap0
        self.rearrange1 = Rearrange(
            "b c h w -> b (h w) c",
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.grapher(x)
        # print(f"Grapher: {x}")
        x = self.rearrange1(x)
        x = self.to_lorentz(x)
        # print(f"Lorentz: {x}")
        x = self.ffn(x)
        # print(f"FFN: {x}")
        x = self.to_euclid(x)
        # print(f"Euclid: {x}")
        x = x.reshape(B, C, H, W)
        # print(x.shape)
        return x


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

    def forward(self, x):
        x = self.convs(x)
        return x


class Lorentz_MLP_head(nn.Module):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, manifold, dim, n_classes, bias):
        super(Lorentz_MLP_head, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.input_dim = dim
        self.output_dim = n_classes
        self.use_bias = bias
        self.cls = ManifoldParameter(
            self.manifold.random_normal((n_classes, dim), std=1.0 / math.sqrt(dim)),
            manifold=self.manifold,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        self.decode_adj = False

    def forward(self, x):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias


class Isotropic_VIG_lorentz(nn.Module):
    def __init__(
        self,
        k: int,
        act: str,
        conv: str,
        drop_path: float,
        blocks: list,
        channels: int,
        dropout: float,
        n_classes: int,
        image_resolution: list,
    ):
        """Initialize isotropic VIG

        Args:
            k (int): neighbor num (default:9)
            act (str): activation layer {relu, prelu, leakyrelu, gelu, hswish}
            norm (str): batch or instance normalization {batch, instance}
            bias (bool): bias of conv layer True or False
            epsilon (float): stochastic epsilon for gcn
            use_stochastic (bool): stochastic for gcn, True or False
            conv (str): graph conv layer {edge, mr}
            drop_path (float): Drop path rate
            blocks (list): number of basic blocks in the backbone
            channels (int): number of channels of deep features
            dropout (float): dropout rate
            n_classes (int): number of output classes.
            image_resolution (list) : Image height and width.
        """
        super(Isotropic_VIG_lorentz, self).__init__()
        self.channels = channels
        self.act = act
        self.stem = Stem(out_dim=self.channels, act=self.act)
        h, w = image_resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, h // 4, w // 4))
        HW = h // 4 * w // 4

        self.k = k
        self.n_blocks = blocks

        dpr = [
            x.item() for x in torch.linspace(0, drop_path, self.n_blocks)
        ]  # stochastic depth decay rule
        num_knn = [
            int(x.item()) for x in torch.linspace(k, k, self.n_blocks)
        ]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(blocks):
            self.backbone += [
                VIG_block_Lorentz(
                    channels=self.channels,
                    k=self.k,
                    dilation=min(idx // 4 + 1, max_dilation),
                    conv=conv,
                    act=act,
                    drop_path=dpr[idx],
                )
            ]
            idx += 1

        self.backbone = Seq(*self.backbone)
        self.prediction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Conv2d(1024, n_classes, 1, bias=True),
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):

        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


class Isotropic_VIG_lorentz_complete(nn.Module):
    def __init__(
        self,
        k: int,
        act: str,
        conv: str,
        drop_path: float,
        blocks: list,
        channels: int,
        dropout: float,
        n_classes: int,
        image_resolution: list,
        manifold: str,
    ):
        """Initialize isotropic VIG

        Args:
            k (int): neighbor num (default:9)
            act (str): activation layer {relu, prelu, leakyrelu, gelu, hswish}
            norm (str): batch or instance normalization {batch, instance}
            bias (bool): bias of conv layer True or False
            epsilon (float): stochastic epsilon for gcn
            use_stochastic (bool): stochastic for gcn, True or False
            conv (str): graph conv layer {edge, mr}
            drop_path (float): Drop path rate
            blocks (list): number of basic blocks in the backbone
            channels (int): number of channels of deep features
            dropout (float): dropout rate
            n_classes (int): number of output classes.
            image_resolution (list) : Image height and width.
        """
        super(Isotropic_VIG_lorentz_complete, self).__init__()
        self.channels = channels
        self.act = act
        self.stem = Stem(out_dim=self.channels, act=self.act)
        h, w = image_resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, h // 4, w // 4))
        HW = h // 4 * w // 4

        self.k = k
        self.n_blocks = blocks

        dpr = [
            x.item() for x in torch.linspace(0, drop_path, self.n_blocks)
        ]  # stochastic depth decay rule
        num_knn = [
            int(x.item()) for x in torch.linspace(k, k, self.n_blocks)
        ]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(blocks):
            self.backbone += [
                VIG_block_Lorentz_complete(
                    channels=self.channels,
                    k=self.k,
                    dilation=min(idx // 4 + 1, max_dilation),
                    conv=conv,
                    act=act,
                    drop_path=dpr[idx],
                )
            ]
            idx += 1

        self.backbone = Seq(*self.backbone)
        self.model_init()
        self.l = Lorentz()
        self.to_euclid = self.l.logmap0
        self.to_lorentz = self.l.expmap0
        self.rearrange = Rearrange("b c h w -> b (h w) c")
        self.lorentz_mlp_head = Lorentz_MLP_head(
            dim=channels, n_classes=n_classes, bias=True, manifold=manifold
        )

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):

        x = self.stem(inputs) + self.pos_embed

        B, C, H, W = x.shape
        x = self.to_lorentz(x)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1)
        print(x.shape)
        x = x.squeeze(-1).squeeze(-1)
        
        x = self.lorentz_mlp_head(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    iso_vig = Isotropic_VIG_lorentz_complete(
        k=9,
        act="gelu",
        conv="lmr",
        drop_path=0.0,
        blocks=16,
        channels=640,
        dropout=0.0,
        n_classes=5,
        image_resolution=[32, 32],
        manifold="Lorentz"
    )
    image_tensor = torch.randn((3, 3, 32, 32))
    output_tensor = iso_vig(image_tensor)
    # print(output_tensor.shape)
    print(output_tensor)
    print(output_tensor.shape)
    print(count_parameters(iso_vig))
