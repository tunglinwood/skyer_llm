import torch
from torch import nn
from torch.nn import init
from rope_transformer import TransformerDecoder


class Skyer(nn.Module):

    # def __init__(self,
    #              num_layers=48,
    #              input_dim=1024,
    #              hide_dim=768,
    #              n_q_heads=8,
    #              n_kv_heads=4,
    #              max_len=1024,
    #              num_vocs=30000):

    def __init__(self,
                 num_layers=20,
                 input_dim=2048,
                 hide_dim=1536,
                 n_q_heads=24,
                 n_kv_heads=12,
                 max_len=1024,
                 num_vocs=30000,
                 cache_max_batch_size=None,
                 cache_max_seq_len=None):
        super().__init__()

        self._cache_max_batch_size = cache_max_batch_size

        self._emb = nn.Embedding(num_vocs, input_dim)

        self._tf_layer = TransformerDecoder(
            num_layers=num_layers,
            input_dim=input_dim,
            hide_dim=hide_dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            max_len=max_len,
            cache_max_batch_size=cache_max_batch_size,
            cache_max_seq_len=cache_max_seq_len
        )

        # self._output_layer = nn.Linear(input_dim,num_vocs,bias=False)
        # self._output_layer.weight = self._emb.weight
        if cache_max_batch_size is not None:
            self.apply(self._init_weight)

    def _init_weight(self,m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
            # m.reset_parameters()
        elif isinstance(m,nn.Embedding):
            init.xavier_uniform_(m.weight)
            # m.reset_parameters()

    
    def _forward(self, ids, start_pos):
        _tokens = self._emb(ids)
        _features = self._tf_layer(_tokens, start_pos)
        return _features@self._emb.weight.T
        # return self._output_layer(_features)


    def forward(self, ids, start_pos=0):
        if self._cache_max_batch_size is None :
            return self._forward(ids, start_pos)
        else:
            with torch.no_grad():
                return self._forward(ids, start_pos)



if __name__ == "__main__":
    skyer = Skyer()

    # x = torch.randint(0, 100, size=(3, 5))
    # skyer(x)
