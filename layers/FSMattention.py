import concurrent.futures
import threading
from torch.nn import Softmax
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
from einops import rearrange, repeat
from layers.Global_manba import global_mamba
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from mosa import MoSA 
class FSMattention(nn.Module): 
    def __init__(self, configs,d_model,in_dim):
        super(FSMattention, self).__init__()

        self.in_projector = nn.Sequential(nn.Linear(d_model, d_model),nn.Dropout(configs.dropout))
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.ones(8, 1))
        self.num_heads = 8
        self.d_k=d_model/8
        self.weight1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(configs.enc_in),
            nn.Dropout(configs.dropout),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.Dropout(configs.dropout),
            nn.Sigmoid())
        self.project_out = nn.Linear(d_model*2, d_model)
        self.down_conv = nn.Sequential(nn.Conv1d(d_model,d_model , 1,padding=1),nn.BatchNorm2d(d_model),
             nn.ReLU(True))
        

    def forward(self, x):
        conv2 = self.in_projector(x)
        b, c, d = conv2.shape # torch.Size([32, 7, 512])
        q=self.W_Q(conv2)
        k=self.W_K(conv2)
        v=self.W_V(conv2)
        q_f_2 = torch.fft.fft2(q.float())
        k_f_2 = torch.fft.fft2(k.float())
        v_f_2 = torch.fft.fft2(v.float())
        tepqkv = torch.fft.fft2(conv2.float())#128

        q_f_2 = rearrange(q_f_2, 'b c (h w) -> b c h w', h=self.num_heads)
        k_f_2 = rearrange(k_f_2, 'b c (h w) -> b c h w', h=self.num_heads)
        v_f_2 = rearrange(v_f_2, 'b c (h w) -> b c h w', h=self.num_heads)
        q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
        k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
        attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature  #torch.Size([32, 7, 8, 8])
        attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
        out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
        out_f_2 = rearrange(out_f_2, 'b c h w -> b c (h w)', h=self.num_heads, b=b, c=c)#torch.Size([32, 7, 512])
        out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight1(tepqkv.real)*tepqkv))  #torch.Size([32, 7, 512])
        out_2 = self.project_out(torch.cat((out_f_2,out_f_l_2),-1))
        F_2 = torch.add(out_2, conv2)
        return F_2
def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)
    normalized_tensor = torch.complex(norm_real, norm_imag)


    return normalized_tensor




class LinearAttention_B(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
 
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)
def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PositionLinearAttention(nn.Module): # torch.Size([32, 7, 512])
    """Position linear attention"""
    def __init__(self, configs, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.query_conv = nn.Linear(configs.d_model, configs.d_model)
        self.key_conv = nn.Linear(configs.d_model, configs.d_model)
        self.value_conv = nn.Linear(configs.d_model, configs.d_model)
        self.l2_norm = l2_norm
        self.eps=1e-6
        self.gamma = Parameter(torch.zeros(1))
 

    def forward(self, x):# torch.Size([32, 7, 512])
        batch_size, chnnels, Lenth = x.shape #torch.Size([32, 7, 512])
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)
        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        tailor_sum = 1 / (Lenth + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, Lenth)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        x=(x + self.gamma * weight_value).contiguous()
        return x