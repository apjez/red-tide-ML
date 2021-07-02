import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Create dataloader for map responses
class RedTideDataset(Dataset):
    def __init__(self, datamatrix, labels):
        self.datamatrix = datamatrix
        self.labels = labels
        
    def __len__(self):
        return self.datamatrix.shape[0]
        
    def __getitem__(self, idx):
        return self.datamatrix[idx, :], self.labels[idx, :]