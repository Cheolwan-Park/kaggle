import torch
from torch.utils import data
import pandas as pd
from PIL import Image
from os import path


class TrainDataset(data.Dataset):
    def __init__(self, transform, datapath):
        super(TrainDataset, self).__init__()

        self.datapath = datapath
        self.data = pd.read_csv(path.join(self.datapath, 'train_labels.csv'))

        self.transform = transform

    def __getitem__(self, idx):
        img_name, label = self.data.iloc[idx]

        img = Image.open(path.join(self.datapath, f'train/{img_name}.tif'))
        img = self.transform(img)

        return img, torch.tensor(label).float()

    def __len__(self):
        return len(self.data)
