import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_channel: int, height: int, width: int):
        super(Generator, self).__init__()

        self.img_channel: int = img_channel
        self.height: int = height
        self.width: int = width

        def block(in_feat: int, out_feat: int, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, img_channel * height * width),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        img: Tensor = self.model(z)
        img = img.view(img.size(0), self.img_channel, self.height, self.width)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_channel: int, height: int, width: int) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_channel * height * width, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: Tensor) -> Tensor:
        img_flat = img.view(img.size(0),  # batch_size
                            -1)
        validity: Tensor = self.model(img_flat)

        # genuine <--> fake
        # 1       <--> 0
        return validity


AdversarialLoss = torch.nn.BCELoss
