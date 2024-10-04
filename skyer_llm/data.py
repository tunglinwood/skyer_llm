from torch.utils.data import Dataset
import torch

class SkyDataset(Dataset):
    '''
        Dataset
    '''

    def __init__(self,file_path,seq):
        _ids = torch.load(file_path)
        _ids = _ids[:_ids.shape[0]//seq*seq]
        self._ids = _ids.reshape(-1,seq)
    
    def __len__(self):
        return self._ids.shape[0]
    
    def __getitem__(self, index):
        return self._ids[index]
