import torch
from torch import nn
from torch.nn import functional as F


def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]


class Attention(nn.Module):

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 cache_max_batch_size,
                 cache_max_seq_len
                 ):
        super().__init__()

        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads

        self._group = n_q_heads // n_kv_heads

        self._head_size = hide_dim // self._n_q_heads

        self._qw = nn.Linear(input_dim, self._head_size*self._n_q_heads)
        self._kw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._vw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._ow = nn.Linear(hide_dim, input_dim)

        self._cache_max_batch_size = cache_max_batch_size
        if self._cache_max_batch_size is not None:
            _cache_k = torch.zeros((cache_max_batch_size,
                                    cache_max_seq_len,
                                    n_kv_heads,
                                    self._head_size,
                                    ))
            self.register_buffer("_cache_k", _cache_k, persistent=False)

            _cache_v = torch.zeros((cache_max_batch_size,
                                    cache_max_seq_len,
                                    n_kv_heads,
                                    self._head_size,
                                    ))
            self.register_buffer("_cache_v", _cache_v, persistent=False)

    def forward(self, x, freq_cis, start_pos):
        _bn, _seq, _ = x.shape

        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[start_pos:start_pos+_seq])
        _k = apply_rotary_emb(_k, freq_cis[start_pos:start_pos+_seq])

        if self._cache_max_batch_size is not None:
            self._cache_k[:_bn, start_pos: start_pos + _seq] = _k
            self._cache_v[:_bn, start_pos: start_pos + _seq] = _v

            _k = self._cache_k[:_bn, : start_pos + _seq]
            _v = self._cache_v[:_bn, : start_pos + _seq]

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        _k = _k[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, start_pos+_seq, self._head_size)
        _v = _v[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, start_pos+_seq, self._head_size)

        _o = F.scaled_dot_product_attention(
            _q, _k, _v, attn_mask=None, is_causal=True)

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)

        return self._ow(_o)


class FFN(nn.Module):

    def __init__(self, input_dim, hide_dim):
        super().__init__()

        self._w0 = nn.Linear(input_dim, hide_dim)
        self._w1 = nn.Linear(input_dim, hide_dim)
        self._w2 = nn.Linear(hide_dim, input_dim)

        self._gate = nn.SiLU()

    def forward(self, x):
        _o0 =self._w0(x)
        _o1 = self._w1(x)
        _g = self._gate(_o1)
        _og = _o0*_g
        return self._w2(_og)


class RMSNormal(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        return self._w*x/((x**2).mean(-1, keepdim=True)**0.5+1e-6)


class TransformerLayer(nn.Module):
    """
    单层的Transformer结构
    """

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 cache_max_batch_size,
                 cache_max_seq_len):
        super().__init__()

        self._att_norm = RMSNormal(input_dim)

        self._att_layer = Attention(input_dim,
                                    hide_dim,
                                    n_q_heads,
                                    n_kv_heads,
                                    cache_max_batch_size,
                                    cache_max_seq_len)

        self._ffn_norm = RMSNormal(input_dim)

        self._ffn_layer = FFN(input_dim,
                              hide_dim)

    def forward(self, x, freq_cis, start_pos):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x, freq_cis, start_pos)

        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y


class TransformerDecoder(nn.Module):
    """
        解码器
    """

    def __init__(self,
                 num_layers,  
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 max_len,
                 cache_max_batch_size=None,
                 cache_max_seq_len=None
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,
                              hide_dim,
                              n_q_heads,
                              n_kv_heads,
                              cache_max_batch_size,
                              cache_max_seq_len) for _ in range(num_layers)]
        )

        _freq_cis = precompute_freqs_cis(hide_dim//n_q_heads, max_len)

        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x, start_pos):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x, self.freq_cis, start_pos)
        return _x

