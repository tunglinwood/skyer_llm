import sentencepiece as sp
import torch
import json
import os
from tqdm import tqdm

class Preprocess:

    def __init__(self,dst_dir):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load("/root/lanyun-fs/team2/team2_model/tokenizer.model")
        self._dst_dir = dst_dir

    def __call__(self,file_path):
        _base_name = os.path.basename(file_path)
        _fn = _base_name.split(".")[0]

        _vocs = []
        for _line in tqdm(open(file_path,"r+",encoding="UTF-8")):
            _txt_js = json.loads(_line)
            _ids = self._spm.Encode(_txt_js["text"])
            _vocs.append(2)
            _vocs.extend(_ids)
            _vocs.append(3)
        _vocs = torch.tensor(_vocs,dtype=torch.int16)
        torch.save(_vocs,f"{self._dst_dir}/{_fn}")


if __name__ == '__main__':
    preprocess = Preprocess("datas")
    preprocess("2022-21_zh_middle_0011.jsonl")