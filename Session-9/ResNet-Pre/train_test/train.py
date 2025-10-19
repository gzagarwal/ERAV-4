from tqdm import tqdm
import torch
import torch.nn as nn
import os
import glob
from typing import Optional, Tuple


def save_checkpoint(
    directory: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **extras,
):
    """Save model + optimizer state to `directory/checkpoint_epoch_{epoch}.pth`.

    extras: any additional scalar metrics or lists (train_losses, etc.) to save.
    """
    os.makedirs(directory, exist_ok=True)
    fname = os.path.join(directory, f"checkpoint_epoch_{epoch}.pth")
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        try:
            state["scheduler_state"] = scheduler.state_dict()
        except Exception:
            pass
    state.update(extras)
    torch.save(state, fname)
    return fname


def get_latest_checkpoint(directory: str) -> Optional[str]:
    """Return path to the latest checkpoint file in `directory` or None."""
    pattern = os.path.join(directory, "checkpoint_epoch_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None

    # sort by epoch parsed from filename
    def epoch_from_name(p):
        base = os.path.basename(p)
        try:
            return int(base.split("_")[-1].split(".")[0])
        except Exception:
            return 0

    files.sort(key=epoch_from_name)
    return files[-1]


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict:
    """Load checkpoint and restore model/optimizer/scheduler if provided.

    Returns the loaded checkpoint dict.
    """
    chk = torch.load(path, map_location="cpu")
    model.load_state_dict(chk["model_state"])
    if optimizer is not None and "optimizer_state" in chk:
        optimizer.load_state_dict(chk["optimizer_state"])
    if scheduler is not None and "scheduler_state" in chk:
        try:
            scheduler.load_state_dict(chk["scheduler_state"])
        except Exception:
            pass
    return chk


def resume_if_available(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """If a checkpoint exists in checkpoint_dir, load it and return next epoch to start from.

    Returns 0 if no checkpoint found or the epoch to resume from (next epoch number).
    """
    latest = get_latest_checkpoint(checkpoint_dir)
    if latest is None:
        return 0
    chk = load_checkpoint(latest, model, optimizer, scheduler)
    start_epoch = chk.get("epoch", 0) + 1
    print(f"Resuming from checkpoint {latest}, starting at epoch {start_epoch}")
    return start_epoch


from torchvision import transforms


import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import torch

# Metric collectors (populated during a run)
train_losses = []
train_acc = []
criterion = nn.CrossEntropyLoss()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MEAN_255 = tuple(int(m * 255) for m in IMAGENET_MEAN)

train_aug = A.Compose(
    [
        A.Resize(256, 256),  # upsample 64 -> 256 first
        A.RandomCrop(224, 224, p=1.0),  # standard ImageNet crop
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03125,  # ~2 px on a 64px base; ok after Resize too
            scale_limit=0.05,
            rotate_limit=8,
            border_mode=0,  # cv2.BORDER_CONSTANT
            value=MEAN_255,  # <-- actually fill with dataset mean
            p=0.7,
        ),
        A.CoarseDropout(
            max_holes=1,
            min_holes=1,
            max_height=16,
            max_width=16,
            min_height=16,
            min_width=16,
            fill_value=MEAN_255,  # fill BEFORE Normalize → use 0–255 scale
            mask_fill_value=None,
            p=0.25,
        ),
        A.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD
        ),  # expects mean/std in 0–1 range
        ToTensorV2(),
    ]
)
test_aug = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


class AlbumentationsWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # img is PIL
        img = np.array(img)  # to HWC uint8
        # Apply Albumentations transform
        if self.transform:
            img = self.transform(image=img)["image"]
        # Convert numpy array (H, W, C) to torch tensor (C, H, W)
        return img, label


def train_transforms(augment: bool = False):
    """Return a torchvision.transforms.Compose for training.

    If augment=True, a small set of lightweight augmentations are added
    (rotation / small translation). These are safe for MNIST and help generalization.
    """
    if augment:
        return transforms.Compose(
            [
                # transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
                transforms.Resize(256),  # upscale tiny images first
                transforms.RandomCrop(224),  # standard ImageNet crop
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


def test_transforms():
    """Return deterministic transforms for evaluation / test sets.

    Usually this is just ToTensor + Normalize. Kept as a helper so callers can
    explicitly use the same normalization as training without accidental augmentations.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train(model, device, train_loader, optimizer, epoch) -> Tuple[float, float]:
    model.train()
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    running_loss = 0.0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        # store a Python float to avoid keeping computational graph
        train_losses.append(loss.item())
        # accumulate loss
        running_loss += loss.item() * data.size(0)  # sum over batch
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}"
        )
        train_acc.append(100 * correct / processed)
    # compute epoch averages
    avg_loss = running_loss / processed
    avg_acc = 100.0 * correct / processed

    return avg_loss, avg_acc
