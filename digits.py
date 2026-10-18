"""MNIST digits 1-9 plus generated blank cells, the data splits, and the three classifiers."""
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import MNIST

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODELS = ["linear", "mlp", "cnn"]
SEEDS = [0, 1, 2]
BLANK_NOISE = 0.05  # standard deviation of the noise in a blank cell (pixels in [0, 1])
VAL_FRACTION = 0.1
SPLIT_SEED = 0
BLANK_SEEDS = {"train": 1, "val": 2, "test": 3}  # cell datasets; grids use their own seeds


def load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MNIST images (N x 1 x 28 x 28, floats in [0, 1]) and labels, with the 0s removed."""
    ds = MNIST(DATA_DIR, train=train, download=True)
    keep = ds.targets != 0
    return ds.data[keep].unsqueeze(1).float() / 255, ds.targets[keep]


def split_indices(n: int, seed: int = SPLIT_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle range(n) with a fixed seed; return training indices (90%) and validation indices (10%)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = round(VAL_FRACTION * n)
    return perm[n_val:], perm[:n_val]


def digit_pool(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the digit images and labels of a split: 'train' and 'val' from MNIST train, 'test' from MNIST test."""
    if split == "test":
        return load_mnist(train=False)
    images, labels = load_mnist(train=True)
    train_idx, val_idx = split_indices(len(labels))
    idx = train_idx if split == "train" else val_idx
    return images[idx], labels[idx]


def make_blanks(n: int, seed: int) -> torch.Tensor:
    """Return n blank cells: black with light Gaussian noise, clipped to [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return (BLANK_NOISE * torch.randn(n, 1, 28, 28, generator=g)).clamp(0, 1)


def cell_dataset(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cells of a split: its digits plus about one class worth of blanks (label 0)."""
    images, labels = digit_pool(split)
    n_blank = round(len(labels) / 9)
    blanks = make_blanks(n_blank, BLANK_SEEDS[split])
    return torch.cat([images, blanks]), torch.cat([labels, torch.zeros(n_blank, dtype=labels.dtype)])


def make_model(name: str) -> nn.Module:
    """Return a fresh classifier ('linear', 'mlp' or 'cnn') mapping N x 1 x 28 x 28 images to N x 10 logits."""
    if name == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 14 x 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64 x 7 x 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )
    raise ValueError(f"unknown model {name!r}")


def model_path(name: str, seed: int) -> Path:
    """Return the file where a trained model is saved."""
    return MODEL_DIR / f"{name}_seed{seed}.pt"


def load_model(name: str, seed: int, device: str = "cpu") -> nn.Module:
    """Load a trained model in evaluation mode."""
    model = make_model(name)
    model.load_state_dict(torch.load(model_path(name, seed), map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model: nn.Module, images: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    """Return the logits of a model on images, computed in batches on the model's device, as a CPU tensor."""
    device = next(model.parameters()).device
    model.eval()
    return torch.cat([model(images[i:i + batch].to(device)).cpu() for i in range(0, len(images), batch)])
