import torch
import torch.nn as nn
import math
from models.gcn_lib.torch_vertex import act_layer
from torch.nn import Sequential as Seq
import torch.nn.functional as F
import hyptorch.nn as hypnn
from models.manifolds.lorentz import Lorentz
import models.manifolds as manifolds
from einops.layers.torch import Rearrange
from models.layers.hyp_layers import LorentzLinear
from geoopt import ManifoldParameter
from fvcore.nn import FlopCountAnalysis
import numpy as np

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


class lorentz_resmlp_block(nn.Module):
    def __init__(
        self,
        manifold: str = "Lorentz",
        channel: int = 768,
        patches: int = 64,
        bias: str = True,
        dropout: float = 0.1,
        scale: int = 10,
        fixscale: bool = False,
        act: str = "gelu",
    ):
        """Lorentz_res_block initialization. Implements cross patch and cross channel lorentz linear layer

        Args:
            manifold (str, optional): Manifold type. Defaults to 'Lorentz'.
            channel (int, optional): Number of channels. Defaults to 768.
            patches (int, optional): Number of patches Defaults to 64.
            bias (str, optional): Boolean for bias. Defaults to True.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            scale (int, optional): Scale value. Defaults to 10.
            fixscale (bool, optional): Whether scale value does not require grad. Defaults to False.
            act (str, optional): Activation function. Defaults to 'gelu'.
        """
        super().__init__()
        self.manifold = manifold
        self.channel = channel
        self.bias = bias
        self.dropout = dropout
        self.scale = scale
        self.fixscale = fixscale
        self.act = act
        self.patches = patches

        # Along the channels
        self.cross_channel_lorentz = LorentzLinear(
            manifold=self.manifold,
            in_features=self.channel,
            out_features=self.channel,
            bias=self.bias,
            dropout=self.dropout,
            scale=self.scale,
            fixscale=self.fixscale,
            nonlin=act_layer(self.act),
        )
        # Along the patches
        self.cross_patch_lorentz = LorentzLinear(
            manifold=self.manifold,
            in_features=self.patches,
            out_features=self.patches,
            bias=self.bias,
            dropout=self.dropout,
            scale=self.scale,
            fixscale=self.fixscale,
            nonlin=act_layer(self.act),
        )

        self.rearrange = Rearrange("b p c -> b c p")
        self.revert_shape = Rearrange("b c p -> b p c")
        self.scale = nn.Parameter(
            torch.ones(()) * math.log(scale), requires_grad=not fixscale
        )

    def scale_func(self, x):
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / (x_narrow * x_narrow).sum(
            dim=-1, keepdim=True
        ).clamp_min(
            1e-8
        )  # Scale function
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def forward(self, x):
        temp = x
        x = self.cross_channel_lorentz(x)
        x = self.rearrange(x)
        x = self.cross_patch_lorentz(x)
        x = self.revert_shape(x)
        x = temp + x
        x = self.scale_func(x)
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


class Lorentz_resmlp(nn.Module):
    def __init__(
        self,
        image_resolution: list,
        num_classes: int = 10,
        num_blocks: int = 12,
        in_dim: int = 3,
        channels: int = 768,
        act: str = "gelu",
        manifold: str = "Lorentz",
        patch_size=8,
    ):
        """Lorentz based ResMLP model.

        Args:
            image_resolution (list): Image size
            num_classes (int, optional): Num classes. Defaults to 10.
            num_blocks (int, optional): Num blocks. Defaults to 12.
            in_dim (int, optional): Input dimension. Defaults to 3.
            channels (int, optional): channel size. Defaults to 768.
            act (str, optional): activation function. Defaults to 'gelu'.
            manifold (str, optional): Manifold. Defaults to 'Lorentz'.
        """
        super().__init__()

        # assert image_resolution[0] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # self.num_patch =  (image_resolution[0]// patch_size) ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_dim, channels, patch_size, patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        self.in_dim = in_dim
        self.channels = channels
        self.act = act
        self.l = Lorentz()
        self.to_euclid = self.l.logmap0
        self.to_lorentz = self.l.expmap0
        self.stem = Stem(in_dim=self.in_dim, out_dim=self.channels, act=self.act)

        h, w = image_resolution
        HW = h // 4 * w // 4
        self.patches = HW

        # self.patches = self.num_patch
        self.num_blocks = num_blocks
        self.manifold = manifold
        self.num_classes = num_classes

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(self.num_blocks):
            self.backbone += [
                lorentz_resmlp_block(channel=self.channels, patches=self.patches)
            ]
            idx += 1

        self.backbone = Seq(*self.backbone)
        self.lorentz_mlp_head = Lorentz_MLP_head(
            dim=self.channels,
            n_classes=self.num_classes,
            bias=True,
            manifold=self.manifold,
        )
        self.rearrange = Rearrange("b p c -> b c p")

    def forward(self, x):
        x = self.stem(x)
        # x = self.to_patch_embedding(x)
        x = self.to_lorentz(x)
        x = self.backbone(x)
        x = self.rearrange(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.lorentz_mlp_head(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    img = torch.randn((3, 3, 32, 32))
    model = Lorentz_resmlp(
        in_dim=3,
        channels=196,
        act="gelu",
        image_resolution=[32, 32],
        patch_size=8,
        num_classes=10,
    )
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    flops = FlopCountAnalysis(model, img)
    print(f"Number of flops: {flops.total()}")
