import torch
import json
import os
from torch.utils.data import Dataset
from PIL import Image


class FlowerDataset(Dataset):
    def __init__(self, root, index_path, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        with open(index_path, 'r') as fp:
            dataset = json.load(fp)

        if train:
            self.filenames = dataset["train"]
        else:
            self.filenames = dataset["test"]

        self.ignore_idxs = []

    def __getitem__(self, idx):
        if idx in self.ignore_idxs:
            return self[idx + 1]

        fpath = os.path.join(self.root, self.filenames[str(idx)])
        try:
            img = Image.open(fpath).convert('RGB')
            data = self.transform(img)
        except Exception as e:
            print(f"Error in loading {self.filenames[str(idx)]}")
            print(str(e))
            self.ignore_idxs.append(idx)
            return self[idx + 1]

        target = int(self.filenames[str(idx)].split('_')[0])  # <label>_<filename>.png
        
        return data, target


    def __len__(self):
        return len(self.filenames)

