import os
import numpy as np
import torch
from torch.utils.data import Dataset

def load_data(data_path='preprocessed'):
    data = []
    labels = []
    for file in os.listdir(data_path):
        if not file.endswith('.npz'):
            continue
        path = os.path.join(data_path, file)
        loaded = np.load(path)
        data.append(loaded['x'])
        labels.append(loaded['y'])
    return data, labels

class SleepDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float).unsqueeze(1), \
               torch.tensor(self.labels[index], dtype=torch.long).unsqueeze(1)


def load_dataset(data_path='preprocessed'):
    data, labels = load_data(data_path)
    return SleepDataset(data, labels)