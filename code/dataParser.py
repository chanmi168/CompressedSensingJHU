from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    # Override to give PyTorch access to any data
    def __getitem__(self, index):
        data = self.data[:,index]
        label = self.labels[index]
        return data, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.data.shape[1]
