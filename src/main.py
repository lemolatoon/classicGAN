import torch
import numpy as np

cuda: bool = True if torch.cuda.is_available() else False


def main():
    print(f"cuda available: {cuda}")


if __name__ == "__main__":
    main()
