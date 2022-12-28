
import torch
import torch.nn as nn
import math
from models.gcn_lib.torch_vertex import DyGraphConv2d, act_layer
from torch.nn import Sequential as Seq
import torch.nn.functional as F
import hyptorch.nn as hypnn
from models.manifolds.lorentz import Lorentz
import models.manifolds as manifolds
from geoopt import ManifoldParameter

def DropPath(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


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


class VIG_block(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilation: int,
        conv: str,
        act: str,
        drop_path: float = 0.0,
    ):
        """VIG block with Grapher and FFN modules

        Args:
            channels (int): Number of channels
            k (int): Number of neighbors
            dilation (int): Dilation size
            conv (str) : Type of convolution
            drop_path (float, optional): Drop rate. Defaults to 0.0.
        """
        super(VIG_block, self).__init__()
        self.grapher = Grapher(
            channels,
            channels,
            k=k,
            dilation=dilation,
            conv=conv,
            act=act,
            drop_path=drop_path,
        )
        self.ffn = FFN(channels, channels, drop_path=drop_path)

    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)
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
    
    
class Isotropic_VIG(nn.Module):
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
            blocks (int): number of basic blocks in the backbone
            channels (int): number of channels of deep features
            dropout (float): dropout rate
            n_classes (int): number of output classes.
            image_resolution (list) : Image height and width.
        """
        super(Isotropic_VIG, self).__init__()
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
                VIG_block(
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
    
    

class Isotropic_VIG_Lorentz_head(nn.Module):
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
            blocks (int): number of basic blocks in the backbone
            channels (int): number of channels of deep features
            dropout (float): dropout rate
            n_classes (int): number of output classes.
            image_resolution (list) : Image height and width.
        """
        super(Isotropic_VIG_Lorentz_head, self).__init__()
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
                VIG_block(
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
        # self.prediction = Seq(
        #     nn.Conv2d(channels, 1024, 1, bias=True),
        #     nn.BatchNorm2d(1024),
        #     act_layer(act),
        #     nn.Dropout(dropout),
        #     nn.Conv2d(1024, n_classes, 1, bias=True),
        # )
        self.lorentz_mlp_head = Lorentz_MLP_head(
            dim=channels, n_classes=n_classes, bias=True, manifold='Lorentz'
        )
        self.l = Lorentz()
        self.to_euclid = self.l.logmap0
        self.to_lorentz = self.l.expmap0
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
        x = x.squeeze(-1).squeeze(-1)
        
        
        return self.lorentz_mlp_head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class poincare_iso_VIG(nn.Module):
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
        c: float,
        ball_dim: int,
        train_x: bool,
        train_c: bool,
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
            c (float) : Curvature of the poincare ball
            ball_dim (int) : Poincare ball dimension
            train_x (bool): Train the exponential map origin
            train_c (bool) : Train the Poincare ball curvature
        """
        super(poincare_iso_VIG, self).__init__()
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
                VIG_block(
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
        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=ball_dim
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=ball_dim, n_classes=10, c=c)
        self.fc1 = nn.Linear(channels * (h // 4) * (w // 4), 1024)
        self.hypl1 = hypnn.HypLinear(1024, ball_dim, c=c)

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
        x = x.reshape(B, -1)

        x = self.fc1(x)
        x = self.tp(x)
        x = self.hypl1(x)
        x = self.mlr(x)
        # x = F.adaptive_avg_pool2d(x, 1)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Isotropic_VIG_Lorentz_head(
        k=9,
        act="gelu",
        conv="mr",
        drop_path=0.0,
        blocks=12,
        channels=196,
        dropout=0.0,
        n_classes=10,
        image_resolution=[32, 32],
        
    )
    ckpt = torch.load('/media/omar/Backup/projects/Fellowship/data/iso_vig_baseline/best.pth')
    pretrained_dict = ckpt["model_state_dict"]
    # model_dict = model.state_dict()
    # # print(model_dict.keys())
    # pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    # model_dict.update(pretrained_dict)
    
    model.load_state_dict(pretrained_dict,strict=False)
    for name, module in model.named_modules():
        
        if 'lorentz_mlp_head' in name:
            # print(name)
            # print(list(module.parameters()))
            module.requires_grad_(requires_grad=True)
            # print(list(module.parameters()))
        else:
            module.requires_grad_(requires_grad=False)
            
    for name, parameters in model.named_parameters():
        if parameters.requires_grad == True:
            print(name)
    print(list(filter(lambda p: p.requires_grad==True, model.parameters())))
    # image_tensor = torch.randn((3, 3, 32, 32))
    # output_tensor = model(image_tensor)
    # print(output_tensor.shape)
    # print(output_tensor)
    print(count_parameters(model))
