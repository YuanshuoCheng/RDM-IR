import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64,bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim//2, bias = bias)
        #self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.to_k = nn.Linear(dim, inner_dim//2, bias = bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is None:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads//2).permute(2, 0, 3, 1, 4)
        k = self.to_k(attn_kv).reshape(B_, N_kv, 1, self.heads, C // self.heads//2).permute(2, 0, 3, 1, 4)
        v = self.to_v(attn_kv).reshape(B_, N_kv, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        return q,k,v

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.q_proj = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1,padding=1)
        self.kv_proj = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1,padding=1)

    def forward(self, x, attn_kv):
        x = self.q_proj(x)
        attn_kv = self.kv_proj(attn_kv)
        bx, cx, hx, wx = x.shape
        x = x.reshape((bx, cx, hx * wx)).transpose(-2, -1)

        bkv, ckv, hkv, wkv = attn_kv.shape
        attn_kv = attn_kv.reshape((bkv, ckv, hkv * wkv)).transpose(-2, -1)

        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        res = x.transpose(-2, -1).reshape(bx, cx, hx, wx)
        return res

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SubAttention(nn.Module):
    def __init__(self,in_channels=128,out_channels=32,bias=False):
        super(SubAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        # self.qk = nn.Conv2d(in_channels=in_channels,out_channels=out_channels*2,
        #                     kernel_size=ratio,stride=ratio,bias=bias)
        self.v = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                           kernel_size=1,stride=1)
    def forward(self,x,qk):# BCHW
        v = self.v(x)
        q = qk[:,self.out_channels:,::]
        k = qk[:,:self.out_channels,::]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=1)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out

class MultiScaleAttention(nn.Module):
    def __init__(self, dim=128,bias=False):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = 4
        self.subatt0 = SubAttention(in_channels=dim, out_channels=dim // 4, bias=bias)
        self.subatt1 = SubAttention(in_channels=dim,out_channels=dim//4,bias=bias)
        self.subatt2 = SubAttention(in_channels=dim, out_channels=dim // 4, bias=bias)
        self.subatt3 = SubAttention(in_channels=dim, out_channels=dim // 4, bias=bias)

        self.to_qk_0 = nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, bias=bias)
        self.to_qk_1 = nn.Conv2d(dim//2, dim // 2, kernel_size=3, padding=1, stride=2, bias=bias)
        self.to_qk_2 = nn.Conv2d(dim//2, dim // 2, kernel_size=3, padding=1, stride=2, bias=bias)
        self.to_qk_3 = nn.Conv2d(dim//2, dim // 2, kernel_size=3, padding=1, stride=2, bias=bias)

        self.project_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,padding=1,stride=1),
            nn.GroupNorm(4,dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )


    def forward(self, x):
        qk_0 = self.to_qk_0(x)
        qk_1 = self.to_qk_1(qk_0)
        qk_2 = self.to_qk_2(qk_1)
        qk_3 = self.to_qk_3(qk_2)
        b, c, h, w = x.shape
        x0 = self.subatt0(x,qk_0)
        x1 = self.subatt1(x,qk_1)
        x2 = self.subatt2(x,qk_2)
        x3 = self.subatt3(x,qk_3)
        out = torch.cat([x0,x1,x2,x3],dim=1)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out



if __name__ == '__main__':
    dim = 128
    q = torch.randn((1,dim,256,256)).cuda()
    kv = torch.randn((1,dim,8,8)).cuda()

    model = CrossAttention(dim=dim, num_heads=4).cuda()
    with torch.no_grad():
        res = model(q,kv)
    print(res.shape)