import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x) # B H/2*W/2 4*C
        x = self.reduction(x)  # B H/2,W/2 2*C

        return x


class SS2D(nn.Module):
    """
    A,D与输入无关，直接通过nn.paramters定义的
    derta，B，C与输入有关，通过linear定义
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 输入特征维度
        self.d_state = d_state  # 状态空间模型的维度
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv # 卷积核大小
        self.expand = expand # 特征扩展倍数
        self.d_inner = int(self.expand * self.d_model)
        # delta矩阵的尺寸是(B, L, d_inner)，但是实际并没有直接映射到d_inner这个维度，而是先映射到dt_rank这么大的维度，
        # 再从dt_rank映射到d_inner，所以dt_rank相当于delta矩阵的秩。好处是dt_rank远小于d_inner，从而降低参数量，默认模式/16不清楚
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # (B,W,H,C)->(B,W,H,4C) 之后要对这层输出依据维度划分为2部分，每部分的维度都是d_inner(d_model的2倍)
        # 为什么->类似于transformer的fnn操作，先将输入映射到一个较大的空间，输出时在对该空间进行压缩
        # m > n：升维，将特征进行各种类型的特征组合，提高模型分辨能力
        # m < n：降维，去除区分度低的组合特征
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 分组卷积层->让每个token预先建立局部的上下文关系，防止独立的token计算
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # 分组卷积，将输入通道和输出通道分组，每两个输入通道对应两个输出通道，最终concat
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs
        )
        self.act = nn.SiLU()

        # 用于生成四个扫描方向的状态空间参数 将输入x映射到状态空间模型的参数Δ、B和C
        # 输出维度：dt_rank：时间常数的低秩表示维度  d_state * 2：B和C矩阵的维度（各占d_state）  这是一种代码友好的方式，之后直接chunk
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        # 初始化时间常数的投影层参数  将Δ从args.dt_rank维度映射到args.d_inner维度
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        # d_state就是hippo矩阵的n的个数，A,D数据独立
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        # 恢复维度
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    # 对时间步参数Δt的投影层初始化
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        # 低秩投影dt_rank << d_inner
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 去找你中初始化为均匀分布
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale  # 保证投影后方差稳定（类似于transformer的1/sqrt(d)的初始化） dt_scale是Δt的初始化振幅
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # 生成指数分布 Δt ~ Exp(Uniform(log(dt_min), log(dt_max)))
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  # dt_init_floor设置最小保护值
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # 逆softplus， 设置为偏置处置
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization  对角线是递增的，与行数有关
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()   # A：[d_inner,d_state]
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        # 将A_log变为一个可学习的参数
        A_log = nn.Parameter(A_log)
        # 权重衰减：防止过拟合
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        # 初始阶段 y ≈ x，模型从简单映射开始学习
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)   # (K,d_inner)
            if merge:
                D = D.flatten(0, 1)
        # 将不可训练的tensor转化为可训练的parameter，并将这个parameter绑定到module里，这样D就变成一个可训练的参数
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True  # 禁用权重衰减，但参数仍会通过梯度正常更新。
        return D

    # 将输入转换为4个方向（原始、转置、翻转、转置翻转），生成(B, K=4, C, H * W)
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape  # C=2*原始C=d_inner
        L = H * W
        K = 4

        # 生成x的原始（水平）和转置（垂直）序列
        # B,C,H,W -> B,2*C,L -> B,2,C,L
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # 添加反向的原始（水平反向）和转置（转置反向）序列
        # B,2,C,L -> B,4,C,L
        # dim=[-1]表示对一个L内部进行翻转，例如：[[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]] -> [[4, 3, 2, 1, 0],[9, 8, 7, 6, 5]]
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        # 投影生成每个扫描方向的dt，B和C的初始值
        # (batch_size, 4方向, dt_rank + 2d_state, 序列长度)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L),   # (batch_size, 4方向, d_inner, 序列长度)
                             self.x_proj_weight)  # (4方向, dt_rank + 2d_state, d_inner)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        # 分割得到三个参数
        # 维度：dts(batch_size, 4方向, dt_rank->d_inner, L)   Bs、Cs：(batch_size, 4方向, d_state, L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        # e^log本质还是A，这里是为了保证A是恒负的(HIPPO矩阵值恒负)
        # 在离散时间系统中，我们希望系统的状态随着时间的推移而衰减或稳定下来，而不是增长，从而避免系统变得不稳定或发散。
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # 原始
        out_y = self.selective_scan(  # 内部应该已经实现了离散化的过程
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,   # 是否使用softplus激活
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # 翻转
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # 转置
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # 转置翻转
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        # projection and 将输入x根据维度分为两个部分
        xz = self.in_proj(x)  # B, H, W, 4C
        x, z = xz.chunk(2, dim=-1) # (b, h, w, 2c)

        x = x.permute(0, 3, 1, 2).contiguous() # (b, 2c, h, w)
        # 先进行卷积，提取局部特征
        x = self.act(self.conv2d(x)) # (b, 2c, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)  # 输出投影
        if self.dropout is not None:
            out = self.dropout(out)
        return out


def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x

class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim//2)
        self.self_attention = SS2D(d_model=hidden_dim//2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        # 上一个分支
        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),  # (B,C/2,H,W)->(B,C/2,H,W)
            nn.Conv2d(in_channels=hidden_dim//2,out_channels=hidden_dim//2,kernel_size=3,stride=1,padding=1), # (B,C/2,H,W)->(B,C/2,H,W)
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2,dim=-1)  # 沿维度方向拆分 (b, h, w, C/2)
        x = self.drop_path(self.self_attention(self.ln_1(input_right))) # 下面的分支
        input_left = input_left.permute(0,3,1,2).contiguous() # (b,c/2,h,w)
        input_left = self.conv33conv33conv11(input_left)  # 上一个分支
        input_left = input_left.permute(0,2,3,1).contiguous() # (b,h,w,c/2)
        output = torch.cat((input_left,x),dim=-1)
        output = channel_shuffle(output,groups=2)
        return output+input


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96,192,384,768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)


        # self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in SS_Conv_SSM, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, SS_Conv_SSM initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0,3,1,2)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.head(x)
        return x


# model = VSSM(num_classes=6).to("cuda")
#
# data = torch.randn(1,3,224,224).to("cuda")
#
# print(model(data).shape)
