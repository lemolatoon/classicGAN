import os
from typing import List
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch


class ImageDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir

        self.idxes: List[int] = []
        self.len: int = 0
        # calculate len
        for f in os.listdir(self.img_dir):
            if f.endswith("jpeg"):
                self.idxes.append(int(f[:-5]))
                self.len += 1
        print(self.len)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.idxes[idx]}.jpeg")
        image = read_image(img_path).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, ()
