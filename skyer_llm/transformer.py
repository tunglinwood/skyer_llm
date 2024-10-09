import torch
from torch import nn
from torch.nn import functional as F


class SkyerFreqsCis:
    def __init__(self):
        pass

    def precompute_freqs_cis(dim, end, theta=50000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class SkyerRotaryEmb:
    def __init__():
        pass

    def apply_rotary_emb(xq, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[:xq_.shape[1]][None, :, None]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq)


class SkyerSdpaAttention(nn.Module):

    def __init__(
        self,
        input_dim,
        hide_dim,
        n_q_heads,
        n_kv_heads,
        cache_max_batch_size,
        cache_max_seq_len,
    ):
        super().__init__()

        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads

        self._group = n_q_heads // n_kv_heads

        self._head_size = hide_dim // self._n_q_heads

        self.q_proj = nn.Linear(input_dim, self._head_size * self._n_q_heads)
        self.k_proj = nn.Linear(input_dim, self._head_size * self._n_kv_heads)
        self.v_proj = nn.Linear(input_dim, self._head_size * self._n_kv_heads)
        self.o_proj = nn.Linear(hide_dim, input_dim, bias=False)
        self.rotary_emb = SkyerRotaryEmb

        self._cache_max_batch_size = cache_max_batch_size
        if self._cache_max_batch_size is not None:
            _cache_k = torch.zeros(
                (
                    cache_max_batch_size,
                    cache_max_seq_len,
                    n_kv_heads,
                    self._head_size,
                )
            )
            self.register_buffer("_cache_k", _cache_k, persistent=False)

            _cache_v = torch.zeros(
                (
                    cache_max_batch_size,
                    cache_max_seq_len,
                    n_kv_heads,
                    self._head_size,
                )
            )
            self.register_buffer("_cache_v", _cache_v, persistent=False)

    def forward(self, x, freq_cis, start_pos):
        _bn, _seq, _ = x.shape

        _q, _k, _v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = self.rotary_emb.apply_rotary_emb(
            _q, freq_cis[start_pos : start_pos + _seq]
        )
        _k = self.rotary_emb.apply_rotary_emb(
            _k, freq_cis[start_pos : start_pos + _seq]
        )

        if self._cache_max_batch_size is not None:
            self._cache_k[:_bn, start_pos : start_pos + _seq] = _k
            self._cache_v[:_bn, start_pos : start_pos + _seq] = _v

            _k = self._cache_k[:_bn, : start_pos + _seq]
            _v = self._cache_v[:_bn, : start_pos + _seq]

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        _k = (
            _k[:, :, None]
            .repeat(1, 1, self._group, 1, 1)
            .reshape(_bn, -1, start_pos + _seq, self._head_size)
        )
        _v = (
            _v[:, :, None]
            .repeat(1, 1, self._group, 1, 1)
            .reshape(_bn, -1, start_pos + _seq, self._head_size)
        )

        _o = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=None, is_causal=True)

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)

        return self.o_proj(_o)


class SkyerMLP(nn.Module):

    def __init__(self, input_dim, hide_dim):
        super().__init__()

        self.gate_proj = nn.Linear(input_dim, hide_dim)
        self.up_proj = nn.Linear(input_dim, hide_dim)
        self.down_proj = nn.Linear(hide_dim, input_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):

        return self.down_proj(self.gate_proj(x)*self.act_fn(self.up_proj(x)))


class SkyerRMSNorm(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, eps=1e-06):
        return self.norm * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+eps)


class SkyerDecoderLayer(nn.Module):
    """
    单层的Transformer结构
    """

    def __init__(
        self,
        input_dim,
        hide_dim,
        n_q_heads,
        n_kv_heads,
        cache_max_batch_size,
        cache_max_seq_len,
    ):
        super().__init__()

        self.input_layernorm = SkyerRMSNorm(input_dim)

        self.self_attn = SkyerSdpaAttention(
            input_dim,
            hide_dim,
            n_q_heads,
            n_kv_heads,
            cache_max_batch_size,
            cache_max_seq_len,
        )

        self.post_attention_layernorm = SkyerRMSNorm(input_dim)

        self.mlp = SkyerMLP(input_dim, hide_dim)

    def forward(self, x, freq_cis, start_pos):
        _x = x
        _x = self.input_layernorm(_x)
        _x = self.self_attn(_x, freq_cis, start_pos)

        _x = x + _x

        _y = _x
        _y = self.post_attention_layernorm(_y)
        _y = self.mlp(_y)

        _y = _y + _x

        return _y


class SkyerModuleList(nn.Module):
    """
    解码器
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        hide_dim,
        n_q_heads,
        n_kv_heads,
        max_len,
        cache_max_batch_size=None,
        cache_max_seq_len=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                SkyerDecoderLayer(
                    input_dim,
                    hide_dim,
                    n_q_heads,
                    n_kv_heads,
                    cache_max_batch_size,
                    cache_max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = SkyerRMSNorm(input_dim)
        self.freq_cis_fn = SkyerFreqsCis

        _freq_cis = self.freq_cis_fn.precompute_freqs_cis(
            hide_dim // n_q_heads, max_len
        )

        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x, start_pos):
        _x = x
        for _layer in self.layers:
            _x = _layer(_x, self.freq_cis, start_pos)
        return self.norm(_x)
