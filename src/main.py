import sys
import traceback
from typing import List, Optional, Tuple, TypedDict, Union, Literal
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import DCDiscriminator, DCGenerator, Generator, Discriminator, AdversarialLoss
from dataset import ImageDataset
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from wandbTypes import sweepConfig, seep_config_with_default
import wandb
import json

# type aliases
Optimizer = torch.optim.Optimizer
Tensor = torch.Tensor

cuda: bool = True if torch.cuda.is_available() else False


def sweepInit():
    print("sweep init")
    print(f"cuda available: {cuda}")
    sweep_config = seep_config_with_default()
    # print(sweep_config, flush=True)
    sweep_id = wandb.sweep(sweep=sweep_config, project="classicDCGAN tune")
    wandb.agent(sweep_id, function=_sweep_entry, count=6)

def main():
    print(f"cuda available: {cuda}")
    batch_size = 32
    resize_rate = 11
    (img_channel, height, width) = (
        3, int(512 / resize_rate), int(768 / resize_rate))
    dir = get_dir("runs")
    print(f"Will be saved in {dir}")
    os.makedirs(dir, exist_ok=True)
    dataloader = getImageDataLoader("images/", height, width,
                                    batch_size=batch_size)
    d_losses, g_losses = train(
        dataloader, img_channel, height, width, root_dir=dir, wandb_enabled=False)
    draw_graph(d_losses, g_losses, batch_size,
               f"{dir}/fig/g_d_loss.png")


def _sweep_entry():
    run = wandb.init()

    print(run)

    config: sweepConfig = wandb.config
    print(config, flush=True)
    # Get DataLoader
    batch_size = config["batch_size"]
    resize_rate = 11
    (img_channel, height, width) = (
        3, int(512 / resize_rate), int(768 / resize_rate))
    dir = get_dir("runs")
    print(f"Will be saved in {dir}")
    os.makedirs(dir, exist_ok=True)
    dataloader = getImageDataLoader("images/", height, width,
                                    batch_size=batch_size)
    d_losses, g_losses = train(
        dataloader, img_channel, height, width, root_dir=dir, sweep_config=config)
    draw_graph(d_losses, g_losses, batch_size,
               f"{dir}/fig/g_d_loss.png")


def train(dataloader: DataLoader, img_channel: int, height: int, width: int, root_dir: str, sweep_config: Optional[sweepConfig] = None, wandb_enabled: bool = False) -> Tuple[List[float], List[float]]:
    print("train started.")
    # loss function
    adversarial_loss = AdversarialLoss()

    # 潜在変数
    latent_dim = 200

    # Initialize generator and discriminator
    generator = DCGenerator(latent_dim, img_channel, height, width)
    discriminator = DCDiscriminator(img_channel, height, width)
    print(next(iter(dataloader))[0].shape)
    if cuda:
        dataloader
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    print("model initialized")

    # Optimizers
    (lr, b1, b2) = (0.0002, 0.5, 0.999)
    if sweep_config is not None:
        lr = sweep_config["lr"]
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(b1, b2))

    n_epoch = 200
    if sweep_config is not None:
        n_epoch = sweep_config["n_epoch"]

    config = {
        "learning_rate": lr,
        "beta_1": b1,
        "beta_2": b2,
        "batch_size": dataloader.batch_size,
        "latent_dim": latent_dim,
        "n_epoch": n_epoch,
        "height": height,
        "width": width,
    }
    with open(f"{root_dir}/config.json", mode="w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if sweep_config is None and wandb_enabled:
        wandb.init("classicGAN", entity="lemolatoon", config=config)
    else:
        # wandb.log({"config": config})
        pass

    sample_interval_per_epoch = 3
    weight_save_interval_per_epoch = 3
    batches_done: int = 0
    d_losses: List[float] = []
    g_losses: List[float] = []
    for epoch in range(n_epoch):

        if epoch % weight_save_interval_per_epoch == 0:
            try:
                dir = f"{root_dir}/weights/"
                os.makedirs(dir, exist_ok=True)
                gen_imgs = train_one_iter(d_losses, g_losses, generator, discriminator,
                                          adversarial_loss, optimizer_G, optimizer_D, dataloader, latent_dim, wandb_enabled, save_model_dir=dir)
            except Exception:
                print(traceback.format_exc())
                print(sys.exc_info()[2])
                print("weight save failed.")
        else:
            gen_imgs = train_one_iter(d_losses, g_losses, generator, discriminator,
                                      adversarial_loss, optimizer_G, optimizer_D, dataloader, latent_dim, wandb_enabled)
        batches_done += len(dataloader)

        print(
            "[Epoch {}/{}] [D loss: {}] [G loss: {}]"
            .format(epoch, n_epoch, d_losses[-1], g_losses[-1])
        )

        if epoch % sample_interval_per_epoch == 0:
            try:
                dir = f"{root_dir}/gen_samples/"
                img_path = f"{dir}{epoch}.png"
                os.makedirs(dir, exist_ok=True)
                save_image(
                    gen_imgs.data[:25], img_path, nrow=5, normalize=True)
                plt.imshow(plt.imread(img_path))
                if wandb_enabled:
                    wandb.log({f"generated_samples.png": plt})
            except:
                print("image save failed.")
    return d_losses, g_losses


def train_one_iter(d_losses: List[float], g_losses: List[float], generator: nn.Module, discriminator: nn.Module, adversial_loss: AdversarialLoss, optimizer_G: torch.optim.Optimizer, optimizer_D: torch.optim.Optimizer, dataloader: DataLoader, latent_dim: int, wandb_enabled: bool, save_model_dir: Optional[str] = None) -> Tensor:
    """
    Return GenImages
    """
    # DataLoader with no label (_)
    for (real_imgs, _) in tqdm(dataloader):
        real_imgs: Tensor
        batch_size = real_imgs.size(0)
        # Adversiarial ground truths
        valid = torch.ones((batch_size, 1), requires_grad=False)
        fake = torch.zeros((batch_size, 1), requires_grad=False)
        if cuda:
            real_imgs = real_imgs.cuda()
            valid = valid.cuda()
            fake = fake.cuda()

        # -----------------
        # Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.from_numpy(np.random.normal(
            0, 1, (real_imgs.shape[0], latent_dim)).astype(np.float32))

        if cuda:
            z = z.cuda()

        # Generate a batch of images
        gen_imgs: Tensor = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss: Tensor = adversial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # -----------------
        # Train Discriminator
        # -----------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversial_loss(discriminator(real_imgs), valid)
        fake_loss = adversial_loss(discriminator(gen_imgs.detach()), fake)
        # NOTE: gen_imgs is `detach`ed from calculation graph of Generator
        d_loss: Tensor = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        d_loss_item = d_loss.item()
        g_loss_item = g_loss.item()
        d_losses.append(d_loss_item)
        g_losses.append(g_loss_item)
        if wandb_enabled:
            wandb.log({
                "d_loss": d_loss_item,
                "g_loss": g_loss_item,
            })

    if save_model_dir is not None:
        try:
            if wandb_enabled:
                wandb.watch((generator, discriminator))
            torch.save(generator.to("cpu").state_dict(),
                       f"{save_model_dir}/generator_latest.pt")
            torch.save(discriminator.to("cpu").state_dict(),
                       f"{save_model_dir}/discriminator_latest.pt")
        except:
            print("weight save failed.")
        if cuda:
            generator.cuda()
            discriminator.cuda()

    return gen_imgs


def getImageDataLoader(img_dir: str, height: int, width: int, batch_size: int = 128,) -> DataLoader:
    transform = transforms.Compose(
        [transforms.Resize((height, width)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])]
    )

    return DataLoader(
        # ImageDataset(img_dir, transform=transform),
        ImageFolder(img_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )


def draw_graph(d_losses: List[float], g_losses: List[float], batch_size: int, save_path: str):
    d_losses = np.array(d_losses)
    g_losses = np.array(g_losses)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d_losses, label="discriminator_loss",
            color="r", linestyle="solid")
    ax.plot(g_losses, label="generator_loss",
            color="g", linestyle="dashed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"batch (batch_size: {batch_size})")
    ax.set_ylabel(f"loss")
    plt.legend(loc="best")
    plt.show()
    plt.savefig(save_path)


def get_dir(root_dir: str, prefix: str = "exp") -> str:
    exp_idx_set = set()
    exp_idx_set.add(0)
    for file_name in os.listdir(root_dir):
        exp_idx_set.add(int(file_name[3:]))
    current_max_idx = max(exp_idx_set)
    return f"{root_dir}/{prefix}{current_max_idx + 1}/"


if __name__ == "__main__":
    # main()
    sweepInit()
