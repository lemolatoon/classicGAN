
import os

import torchvision

from tqdm import tqdm
from models import DCGenerator
import numpy as np
import torch
from main import Tensor, align_to, get_dir, getCachedImageDataSetList
from torchvision.utils import make_grid, save_image


cuda: bool = True if torch.cuda.is_available() else False
def main():
    resize_rate = 11
    process_image_size = lambda x: align_to(int(x / resize_rate), 8)
    (img_channel, height, width) = (
        3, process_image_size(512), process_image_size(768))
    latent_dim = 200
    dataset = getCachedImageDataSetList("images/", height, width,
                                    batch_size=1024, path_pkl=f"./src/pkl_cache/dataset_{height}_{width}_{1024}.pkl")
    save_image(dataset[0][:25], "real_images.png", nrow=5)
    model = DCGenerator(latent_dim, img_channel, height, width)
    model.load_state_dict(torch.load("runs/exp120/weights/generator_latest.pt"))
    dir = get_dir("vals")
    os.makedirs(dir, exist_ok=True)

    # Sample noise as generator input
    z = torch.from_numpy(np.random.normal(
        0, 1, (25, latent_dim)).astype(np.float32))
    # vector = torch.from_numpy(np.random.normal(
    #     0, 0.01, (25, latent_dim)).astype(np.float32))
    vector = torch.fill(z.clone(), 1 / 3000)

    if cuda:
        model.cuda()
    
    frames = []
    for n_iter in tqdm(range(3000)):
        latent_vector = torch.remainder(z + n_iter * vector, 1)
        if cuda:
            latent_vector = latent_vector.cuda()
        img: Tensor = model(latent_vector)
        
        img: Tensor = make_grid(img, nrow=5, normalize=True)
        save_image(img, "generated_images.png", nrow=5)
        frames.append(torchvision.transforms.ToPILImage()(img.cpu()))

    frames[0].save(f"{dir}/gif.gif", save_all=True, append_images=frames[1:], optimize=False, duration=10, loop=0)
    


if __name__ == "__main__":
    main()
