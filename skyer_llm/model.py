import torch
from torch import nn
from torch.nn import init
from skyer_llm.transformer import SkyerModuleList

class Skyer(nn.Module):

    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 hide_dim: int,
                 n_q_heads: int,
                 n_kv_heads: int,
                 max_len: int,
                 num_vocs: int,
                 cache_max_batch_size: int = None,
                 cache_max_seq_len: int = None):
        super().__init__()

        self._cache_max_batch_size = cache_max_batch_size

        self.embed_tokens = nn.Embedding(num_vocs, input_dim)

        self.layers = SkyerModuleList(
            num_layers=num_layers,
            input_dim=input_dim,
            hide_dim=hide_dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            max_len=max_len,
            cache_max_batch_size=cache_max_batch_size,
            cache_max_seq_len=cache_max_seq_len
        )

        if cache_max_batch_size is not None:
            self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight)

    def _forward(self, ids, start_pos):
        _tokens = self.embed_tokens(ids)    
        _features = self.layers(_tokens, start_pos)
        return _features@self.embed_tokens.weight.T

    def forward(self, ids, start_pos=0):
        if self._cache_max_batch_size is None:
            return self._forward(ids, start_pos)
        else:
            with torch.no_grad():
                return self._forward(ids, start_pos)
