import torch
import sentencepiece as spm
from skyer_llm.model import Skyer
from torch.distributions.categorical import Categorical



class Inference:

    def __init__(self,
                 topk=4,
                 temp=20,
                 model_path = "",
                 tokenizer_path = "",
                 ):

        self._topk = topk
        self._temp = temp

        self._skyer = Skyer(num_layers=20,
                            input_dim=2048,
                            hide_dim=1536,
                            n_q_heads=24,
                            n_kv_heads=12,
                            max_len=1024,
                            num_vocs=30000,
                            cache_max_batch_size=1,
                            cache_max_seq_len=1024).cuda()
        self._skyer.eval()
        self._skyer.load_state_dict(torch.load(model_path), strict=False)

        self._spm = spm.SentencePieceProcessor(tokenizer_path)
        self._spm.Load()

    def __call__(self,prompt):
        _vocs = prompt
        _prompt_ids = self._spm.Encode(prompt)
        _ids = torch.tensor(_prompt_ids, dtype=torch.long)[None].cuda()
        _id,_voc = self.forward(_ids,0)
        _vocs+=_voc
        _start_pos = _ids.shape[1]

        for _ in range(10):
            _id,_voc = self.forward(_id,_start_pos)
            _start_pos+=1
            _vocs+=_voc
        return _vocs
    
    def forward(self, ids,start_pos):
        _os= self._skyer(ids,start_pos)
        _o = _os[:,-1]
        _weight, _indices = torch.topk(_o, self._topk, dim=-1)
        _probs = self._tsoftmax(_weight,self._temp)
        # _m = Categorical(_probs)
        # _s = _m.sample()
        _s = torch.multinomial(_probs,1)
        _id = torch.gather(_indices,dim=-1,index=_s)
        return _id,self._spm.Decode(_id.item())

    @staticmethod
    def _tsoftmax(xs,temp=1.):
        _o = xs-xs.mean()
        return torch.exp(_o/temp)/(torch.exp(_o/temp).sum(-1)+1e-5)
    

<<<<<<< HEAD
=======
    
if __name__ == '__main__':
    model_path = "out/model.pt"#model的路径
    tokenizer_path = "out/tokenizer.model"#tokenizer的路径
    env=Inference()
    voc = env("我爱天安门")
    print(voc)
>>>>>>> 34dbad5c20e3af3ed25fa18d14e1e61f8dd02707
