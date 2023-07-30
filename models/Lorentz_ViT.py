import torch
import torch.nn as nn
import math
from models.gcn_lib.torch_vertex import act_layer
from torch.nn import Sequential as Seq
from models.manifolds.lorentz import Lorentz
import models.manifolds as manifolds
from einops.layers.torch import Rearrange
from models.layers.hyp_layers import LorentzLinear
from geoopt import ManifoldParameter
import torch.nn.functional as F


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


class Lorentz_resmlp_block(nn.Module):
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


class Lorentz_centroid(nn.Module):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, manifold, dim, n_classes, bias):
        super(Lorentz_centroid, self).__init__()
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


class Lorentz_attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        manifold="Lorentz",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.tensor([math.sqrt(head_dim)]))

        self.qkv = LorentzLinear(
            in_features=dim,
            out_features=dim * 3,
            bias=qkv_bias,
            dropout=0.0,
            manifold=manifold,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LorentzLinear(
            in_features=dim, out_features=dim, manifold=manifold, dropout=proj_drop
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Lorentz_VIT_Block(nn.Module):
    def __init__(self, dim, manifold, patches, scale, fixscale: bool = False):
        super().__init__()
        self.MLP = Lorentz_resmlp_block(manifold=manifold, channel=dim, patches=patches)
        self.attention = Lorentz_attention(dim=dim)
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
        x = self.attention(x)
        x = temp + x
        x = self.scale_func(x)
        x = self.MLP(x)
        return x


class Lorentz_ViT(nn.Module):
    def __init__(
        self,
        image_resolution: list,
        num_classes: int = 10,
        num_blocks: int = 12,
        in_dim: int = 3,
        channels: int = 768,
        act: str = "gelu",
        manifold: str = "Lorentz",
        dropout: float = 0.1,
        scale: int = 10,
    ):
        super().__init__()
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
        self.num_blocks = num_blocks
        self.manifold = manifold
        self.num_classes = num_classes
        self.dropout = dropout

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(self.num_blocks):
            self.backbone += [
                Lorentz_VIT_Block(
                    dim=self.channels,
                    manifold=self.manifold,
                    patches=self.patches,
                    scale=scale,
                )
            ]
            idx += 1

        self.backbone = Seq(*self.backbone)
        self.lorentz_centroid = Lorentz_centroid(
            dim=self.channels,
            n_classes=self.num_classes,
            bias=True,
            manifold=self.manifold,
        )
        self.rearrange = Rearrange("b p c -> b c p")

    def forward(self, x):
        x = self.stem(x)
        # o = torch.zeros_like(x)
        # x = torch.cat([o[:,0:1,:],x],dim=1)
        x = self.to_lorentz(x)
        x = self.backbone(x)
        x = self.rearrange(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.lorentz_centroid(x)
        return x


if __name__ == "__main__":
    image_tensor = torch.randn((3, 3, 32, 32))
    model = lorentz_resmlp = Lorentz_ViT(
        in_dim=3,
        channels=192,
        act="gelu",
        image_resolution=[32, 32],
        num_blocks=12,
    )

    output = model(image_tensor)
    print(output.shape)
    print(output)
