import os

import pandas as pd
import torch
from PIL import Image


def loader(image_path):
    with open(image_path, 'rb') as f:
        image = Image.open(f).convert('RGB')
    return image


class LPCVDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms):
        _df = pd.read_csv(csv_file)
        self.images = [os.path.join(root, x) for x in _df['images']]
        self.targets = _df['targets']
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transforms(loader(self.images[idx]))
        target = self.targets[idx]

        return img, target
