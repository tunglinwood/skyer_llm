"""
A module for loading sequences of data from a file into a PyTorch Dataset.

This module contains the SkyDataset class, which reshapes data into sequences 
of a fixed length for training deep learning models.
"""

from torch.utils.data import Dataset
import torch

class SkyDataset(Dataset):
    """
    A custom dataset for loading sequences of data from a file.

    This dataset reshapes the data into sequences of a fixed length.

    :param file_path: Path to the file containing the dataset.
    :type file_path: str
    :param seq: The sequence length to split the data into.
    :type seq: int
    """

    def __init__(self, file_path:str, seq:int):
        """
        Initialize the dataset by loading data from the file and reshaping it.

        :param file_path: Path to the file containing the dataset.
        :type file_path: str
        :param seq: The sequence length to split the data into.
        :type seq: int
        """
        _ids = torch.load(file_path)
        _ids = _ids[:_ids.shape[0] // seq * seq]
        self._ids = _ids.reshape(-1, seq)

    def __len__(self):
        """
        Return the sequence length in the dataset.

        :return: The sequence length.
        :rtype: int
        """
        return self._ids.shape[0]

    def __getitem__(self, index):
        """
        Get a specific sequence by index.

        :param index: The index of the sequence to retrieve.
        :type index: int
        :return: The sequence at the specified index.
        :rtype: torch.Tensor
        """
        return self._ids[index]
