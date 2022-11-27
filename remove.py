from torchvision.io import read_image
import os
from tqdm import tqdm
import asyncio


async def main():
    dir = "images/1/"
    tasks = []
    for f in tqdm(os.listdir(dir)):
        if f.endswith("jpeg"):
            tasks.append(asyncio.create_task(check_file(f"{dir}{f}")))
    await asyncio.gather(*tasks)


async def check_file(path):
    try:
        read_image(path)
    except:
        # 画像ファイルとして破損しているものを削除
        print(path)
        os.remove(path)


if __name__ == "__main__":
    asyncio.run(main())
