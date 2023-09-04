import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
import hyptorch.nn as hypnn
from models.gcn_lib.torch_vertex import act_layer


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

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MLPblock(nn.Module):
    def __init__(self, dim, num_patch, mlp_dim, dropout=0.0, init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange("b n d -> b d n"),
            nn.Linear(num_patch, num_patch),
            Rearrange("b d n -> b n d"),
        )
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x


class ResMLP(nn.Module):
    def __init__(
        self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))

        self.affine = Aff(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x):

        x = self.to_patch_embedding(x)

        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)


class Poincare_ResMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        dim,
        num_classes,
        patch_size,
        image_size,
        depth,
        mlp_dim,
        c,
        train_x,
        train_c,
        ball_dim,
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))

        self.affine = Aff(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))
        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=ball_dim
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=ball_dim, n_classes=10, c=c)

        self.hypl1 = hypnn.HypLinear(dim, ball_dim, c=c)

    def forward(self, x):

        x = self.to_patch_embedding(x)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)

        x = x.mean(dim=1)

        x = self.tp(x)
        x = self.hypl1(x)
        x = self.mlr(x)

        return x
    
class Lor_ResMLP_ablation(nn.Module):
    def __init__(
        self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.in_channels = in_channels
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = Stem(in_dim=self.in_channels, out_dim=dim, act="gelu")

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))

        self.affine = Aff(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x):

        x = self.to_patch_embedding(x)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)
    


if __name__ == "__main__":
    img = torch.ones([1, 3, 32, 32])

    model = Lor_ResMLP_ablation(
        in_channels=3,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=196,
        depth=12,
        mlp_dim=196
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    out_img = model(img)
    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
