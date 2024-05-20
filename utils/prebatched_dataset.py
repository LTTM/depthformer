import numpy as np
import os
import torch
from torch.utils.data import Dataset

class PrebatchedDataset(Dataset):
    def __init__(self, root_path, extension="npz"):
        self.items = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f)) and f.endswith(extension)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        d = np.load(self.items[index])
        x, y = d["x"].astype(np.float32)-127.5, d["y"].astype(np.long)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y