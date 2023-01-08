import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

class DCGenerator(nn.Module):
    def __init__(self, latent_dim: int, img_channel: int ,height: int, width: int):
        super(DCGenerator, self).__init__()

        # 2 times `Upsample`
        self.height = height // 4
        self.width = width // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.height * self.width))

        # each `Conv2d` retains its shape like
        # (N, C, H, W) -> (N, C, H, W)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        out: Tensor = self.l1(z)
        out = out.view(out.shape[0], 128, self.height, self.width)
        img: Tensor = self.conv_blocks(out)
        return img

class DCDiscriminator(nn.Module):
    def __init__(self, img_channel: int, height: int, width: int):
        super(DCDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))

            return block
        self.img_channel = img_channel
        self.height = height
        self.width = width
        self.model = nn.Sequential(
            *discriminator_block(self.img_channel, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # calc shape
        with torch.no_grad():
            out: Tensor = self.model(torch.zeros((1, self.img_channel, self.height, self.width)))
        self.adv_layer = nn.Sequential(nn.Linear(out.shape[1] * out.shape[2] * out.shape[3], 1), nn.Sigmoid())

    def forward(self, img: Tensor):
        out: Tensor = self.model(img)
        # `out.shape[0]` is batch size
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# expect (height: 46, width: 69, img_channel: 3)
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
