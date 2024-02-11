from einops import rearrange
from functools import partial
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

import math

# constants (change 2.8.2023)

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig',
                                      ['enable_flash', 'enable_math', 'enable_mem_efficient'])


@dataclass
class Intermediates:
    qk_similarities: Tensor = None
    pre_softmax_attn: Tensor = None
    post_softmax_attn: Tensor = None


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


class FlashAttention(nn.Module):
    def __init__(
            self,
            causal=False,
            dropout=0.,
            flash=True
    ):
        super().__init__()

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse(
            '2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def flash_attn(
            self,
            q, k, v,
            mask=None,
            attn_bias=None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention
        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=causal
            )

            return out

    def forward(self, q, k, v, mask=None, attn_bias=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


# class FlashMHA(nn.Module):
#     def __init__(
#             self,
#             embed_dim,
#             num_heads,
#             window_size,
#             dropout=0.0,
#             device=None,
#             dtype=None
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#
#         self.head_dim = embed_dim // num_heads
#         self.scaling = self.head_dim ** -0.5
#
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         # 相对位置偏置表
#         # 2*8-1 > 15*15  => 225
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
#
#         # self.relative_position_bias_table = nn.Parameter(
#         #     torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
#         #                 864,
#         #                 # num_heads
#         #                 ))  # 2*Wh-1 * 2*Ww-1, nH,
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         # 是计算了窗口内每个位置与其他位置的相对位置。这一步利用广播机制
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer('relative_position_index', relative_position_index)
#
#         # self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
#         self.outproj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout_module = torch.nn.Dropout(dropout).to(device=device, dtype=dtype)
#
#         # init flash attention
#         self.flash_attention = FlashAttention(dropout=dropout, heads=num_heads, scale=self.scaling)
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
#         nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
#         nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         nn.init.constant_(self.out_proj.bias, 0.0)
#
#     def flops(self, n):
#         # calculate flops for 1 window with token length of n
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += n * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * n * (self.dim // self.num_heads) * n
#         #  x = (attn @ v)
#         flops += self.num_heads * n * n * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += n * self.dim * self.dim
#         return flops
#
#     def forward(
#             self,
#             x,
#             mask=None,
#             attn_mask=None,
#             incremental_state=None,
#             is_first_step=False
#     ):
#         # 表示新的 batch 维度大小，即 batch_size
#         # 表示新的序列长度维度大小，即 sequence_length。
#         # 表示每个头中的特征维度大小
#
#         b_, n, c = x.shape
#
#         tgt_len = n
#         bsz = b_
#         # 144 ,6, 64, 30
#         # batch_size, 6个头部自空间  ,sequence_length， 每个头部自空间为30
#         #  6, 30  <= c // self.num_heads
#         qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         # q = self.q_proj(x)
#         # k = self.k_proj(x)
#         # v = self.v_proj(x)
#
#         q *= self.scaling
#         # 64 ,864, 30
#         q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
#         k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
#         v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
#
#         if mask is not None:
#             mask = mask.unsqueeze(1).expand(-1, tgt_len, -1)
#             mask = mask.view(mask.size(0) * self.num_heads, tgt_len, -1)
#
#         # q =  torch.Size([1944, 64, 30])
#         attn_weights = (q @ k.transpose(-2, -1))
#         # torch.Size([1944, 64, 64])
#         attn_weights, _ = self.flash_attention(q, k, v, mask)
#         # assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, k.size(1)]
#
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1],
#                                                         self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         # 225 * 864
#         attn = attn_weights + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nw = mask.shape[0]
#             attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, n, n)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#         attn = self.dropout_module(attn)
#         # attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
#         x = self.outproj(x)
#         # x = self.proj_drop(x)
#         return x
#
#         # attn = torch.bmm(attn_weights, v)
#         # assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
#         # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
#         # attn = self.out_proj(attn)
#
#         # return attn, attn_weights

class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(dropout=dropout, causal=causal)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, query, key, value):
        qkv = self.Wqkv(query)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b s h d', three=3, h=self.num_heads, d=self.head_dim).unbind(
            dim=0)
        context = self.inner_attn(q, k, v)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


# class RelativePositionBias(nn.Module):
#     def __init__(
#             self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12
#     ):
#         super().__init__()
#         self.bidirectional = bidirectional
#         self.num_buckets = num_buckets
#         self.max_distance = max_distance
#         self.n_heads = n_heads
#         self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)
#
#     @staticmethod
#     def _relative_position_bucket(
#             relative_position, bidirectional=True, num_buckets=32, max_distance=128
#     ):
#         ret = 0
#         n = -relative_position
#         if bidirectional:
#             num_buckets //= 2
#             ret += (n < 0).to(torch.long) * num_buckets
#             n = torch.abs(n)
#         else:
#             n = torch.max(n, torch.zeros_like(n))
#
#         max_exact = num_buckets // 2
#         is_small = n < max_exact
#
#         val_if_large = max_exact + (
#                 torch.log(n.float() / max_exact)
#                 / math.log(max_distance / max_exact)
#                 * (num_buckets - max_exact)
#         ).to(torch.long)
#         val_if_large = torch.min(
#             val_if_large, torch.full_like(val_if_large, num_buckets - 1)
#         )
#
#         ret += torch.where(is_small, n, val_if_large)
#         return ret
#
#     def compute_bias(self, qlen, klen, step=None):
#         step = 0 if step is None else step
#         context_position = torch.arange(
#             step,
#             step + qlen,
#             dtype=torch.long,
#             device=self.relative_attention_bias.weight.device,
#         )[:, None]
#         memory_position = torch.arange(
#             klen, dtype=torch.long, device=self.relative_attention_bias.weight.device
#         )[None, :]
#         relative_position = memory_position - context_position  # shape (qlen, klen)
#
#         rp_bucket = self._relative_position_bucket(
#             relative_position,  # shape (qlen, klen)
#             bidirectional=self.bidirectional,
#             num_buckets=self.num_buckets,
#             max_distance=self.max_distance,
#         )
#         rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
#         values = self.relative_attention_bias(
#             rp_bucket
#         )  # shape (qlen, klen, num_heads)
#         values = values.permute([2, 0, 1]).unsqueeze(
#             0
#         )  # shape (1, num_heads, qlen, klen)
#         return values
#
#     def forward(self, batch_size, qlen, klen, step=None):
#         # shape (batch * num_heads, qlen, klen)
#         return (
#             self.compute_bias(qlen, klen, step)
#             .repeat(batch_size, 1, 1, 1)
#             .view(-1, qlen, klen)
#         )


# class XPOS(nn.Module):
#     def __init__(
#             self, head_dim, scale_base=512
#     ):
#         super().__init__()
#         self.head_dim = head_dim
#         self.scale_base = scale_base
#         self.register_buffer(
#             "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
#         )
#
#     def forward(self, x, offset=0, downscale=False):
#         length = x.shape[2]
#         min_pos = -(length + offset) // 2
#         max_pos = length + offset + min_pos
#         scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
#         sin, cos = fixed_pos_embedding(scale)
#
#         if scale.shape[0] > length:
#             scale = scale[-length:]
#             sin = sin[-length:]
#             cos = cos[-length:]
#
#         if downscale:
#             scale = 1 / scale
#
#         # x = apply_rotary_pos_emb(x, sin, cos, scale)
#         sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
#         # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
#         return (x * cos) + (rotate_every_two(x) * sin)
#         return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input tensor
        x = x + self.pe[:x.size(0), :]
        return x


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


class CBAMAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size

        # 通道注意力部分
        self.channel_attention = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.SiLU()
        )

        # 空间注意力部分
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2),
            nn.SiLU()
        )

        # self.front_attention = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // reduction_ratio, 3, 1, 1),
        #     nn.Mish(),
        #     nn.Conv2d(in_channels // reduction_ratio, in_channels, 3, 1, 1),
        # )

    def forward(self, x, x_size):
        x = blc_to_bchw(x, x_size).contiguous()
        if x.size(1) != self.in_channels:
            x = x.view(-1, self.in_channels, self.spatial_kernel_size, self.spatial_kernel_size)

        # x = self.front_attention(x)

        # 通道注意力
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        # 空间注意力
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return bchw_to_blc(x)


import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    def __init__(self, window_size):
        super(RelativePositionBias, self).__init__()
        self.window_size = window_size

        # Compute relative position bias table
        relative_coords_h = torch.arange(self.window_size[0])
        relative_coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))  # Wh*Ww, Wh*Ww

        # Initialize relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))

    def forward(self, attn_weights):
        # Get relative position bias based on attention weights
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(attn_weights.size(0), -1,
                                                             attn_weights.size(2))  # b, Wh*Ww, nH

        # Add relative position bias to attention weights
        attn_weights = attn_weights + relative_position_bias.unsqueeze(1)

        return attn_weights


class LongformerSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, attention_window=8, dropout_prob=0.1):
        super(LongformerSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_window = attention_window
        self.dropout = nn.Dropout(dropout_prob)

        # Define linear projections for queries, keys, and values
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Define global attention bias
        self.global_attention_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Project queries, keys, and values
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Apply global attention bias
        attention_scores += self.global_attention_bias

        # Mask attention scores for the attention window
        attention_mask = self.get_attention_mask(seq_len, self.attention_window)
        attention_scores += attention_mask

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        context = torch.matmul(attention_weights, value)

        # Reshape and concatenate to get the final output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return context

    def get_attention_mask(self, seq_len, attention_window):
        # Create an attention mask to restrict attention to the window
        mask = torch.zeros(1, 1, seq_len, seq_len, device="cuda")
        mask[:, :, :attention_window, :attention_window] = float("-inf")
        mask[:, :, -attention_window:, -attention_window:] = float("-inf")
        return mask


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()

        # 定义权重学习模块
        self.W1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.W2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x1, x2):
        # batch_size, seq_len, embed_dim = x.size()
        # 计算注意力分数
        attn_scores1 = self.W1(x1.permute(3,))
        attn_scores2 = self.W2(x2)

        # 计算注意力权重
        attn_weights1 = torch.softmax(attn_scores1, dim=1)
        attn_weights2 = torch.softmax(attn_scores2, dim=1)

        # 使用注意力权重融合特征
        fused_x1 = x1 * attn_weights1
        fused_x2 = x2 * attn_weights2

        # 将融合后的特征相加或者拼接，这取决于你的需求
        # 这里简单地相加
        fused_output = fused_x1 + fused_x2

        return fused_output


#
# class DilatedSelfAttention(nn.Module):
#     def __init__(self, in_channels, num_heads, dilation_factor=4):
#         super(DilatedSelfAttention, self).__init__()
#         self.in_channels = in_channels
#         self.num_heads = num_heads
#         self.dilation_factor = dilation_factor
#
#         self.query = nn.Linear(in_channels, in_channels)
#         self.key = nn.Linear(in_channels, in_channels)
#         self.value = nn.Linear(in_channels, in_channels)
#
#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#
#         # 膨胀操作，增大了每个位置的注意力范围
#         dilated_seq_length = seq_length * self.dilation_factor
#
#         # 计算query、key和value
#         q = self.query(x).view(batch_size, seq_length, self.num_heads, -1)
#         k = self.key(x).view(batch_size, seq_length, self.num_heads, -1)
#         v = self.value(x).view(batch_size, seq_length, self.num_heads, -1)
#
#         # 膨胀操作，增大了每个位置的注意力范围
#         dilated_k = torch.cat([k[:, i::self.dilation_factor] for i in range(self.dilation_factor)], dim=1)
#
#         # 计算注意力权重
#         attn_weights = torch.einsum('bhnd,bhkd->bhnk', q, dilated_k)
#         attn_weights = attn_weights / (k.size(-1) ** 0.5)
#
#         # 使用softmax计算注意力权重
#         attn_weights = torch.softmax(attn_weights, dim=-1)
#
#         # 计算加权平均后的value
#         attn_output = torch.einsum('bhnk,bhkd->bhnd', attn_weights, v)
#         attn_output = attn_output.contiguous().view(batch_size, -1, self.in_channels)
#
#         return attn_output


class DilatedAttention(nn.Module):
    """
    Dilated Attention Module.

    Arguments:
        d_model: The dimension of the attention layers.
        num_heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        casual (optional): If set to True, the attention mechanism is casual. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """

    def __init__(self, d_model, num_heads, dilation_rate, segment_size, window_size, dropout=0.0, casual=False,
                 use_xpos=False,
                 use_rel_pos_bias=False):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        self.attention = FlashAttention(causal=self.casual, dropout=dropout, flash=False)
        # >>>>>>>>
        self.window_size = window_size

        if use_xpos:
            # self.xpos = XPOS(head_dim=d_model // num_heads)
            self.xpos = PositionalEncoding(embed_dim=d_model)
        if use_rel_pos_bias:
            # self.relative_bias = RelativePositionBias(num_buckets=64, max_distance=256, n_heads=num_heads)
            self.relative_bias = RelativePositionBias(segment_size=segment_size, d_model=d_model)

        # head offsets
        self.head_offsets = nn.Parameter(torch.randn(num_heads, d_model))

    def get_mask(self, i, j):
        return torch.ones((i, j), dtype=torch.bool).triu(j - i + 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        if self.use_xpos:
            x = self.xpos(x)

        # Split and sparsify
        # torch.Size([144, 64, 192])
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        #  torch.Size([144, 1, 64, 192])
        # x = x[:, :, :: self.dilation_rate, :]

        # Perform attention
        attn_output = self.attention(x, x, x)

        # if use rel pos => apply relative positioning bias
        if self.use_rel_pos_bias:
            attn_output = self.relative_bias(attn_output)

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))
            attn_output = attn_output.masked_fill(mask, float('-inf'))

        # attn_output = self.softmax(attn_output)
        # # apply dropout
        # attn_output = self.dropout(attn_output)

        # Scatter and concatenate
        # torch.Size([144, 1, 64, 192])
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)
        # torch.Size([144, 64, 192])

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output
