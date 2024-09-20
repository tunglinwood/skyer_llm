from torch.utils.data import Dataset
import torch
import warnings
import sentencepiece as sp

warnings.filterwarnings("ignore")

class MyDataset(Dataset):

    def __init__(self,file_path,seq):
        _ids = torch.load(file_path)
        _ids = _ids[:_ids.shape[0]//seq*seq]
        self._ids = _ids.reshape(-1,seq)
    
    def __len__(self):
        return self._ids.shape[0]
    
    def __getitem__(self, index):
        return self._ids[index]

if __name__ == "__main__":
    dataset = MyDataset("/root/lanyun-fs/team2/team2_data/token/wanjuan_part-000467-a894b46e", 128)
    print(dataset[0])
    spm = sp.SentencePieceProcessor()
    spm.Load("/root/lanyun-fs/team2/team2_model/tokenizer.model")
    print(spm.Decode(dataset[0].tolist()))