from tqdm import tqdm
import torch.nn as nn

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

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

MEAN_255 = tuple(int(m * 255) for m in CIFAR10_MEAN)

train_aug = A.Compose(
    [
        A.RandomCrop(32, 32, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03125,  # up to ~2 px on 32x32
            scale_limit=0.05,
            rotate_limit=8,
            border_mode=0,  # constant fill
            value=None,  # fill border with dataset mean
            p=0.7,
        ),
        # NOTE: Your params listed max_width=1 but min_width=16 — likely a typo.
        # Using max_width=16 to match min/max pairs for a 16x16 patch.
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=MEAN_255,  # drop with dataset-mean color
            mask_fill_value=None,  # fine for classification (no mask)
            p=0.25,
        ),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
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


test_aug = A.Compose(
    [
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ]
)


def train_transforms(augment: bool = False):
    """Return a torchvision.transforms.Compose for training.

    If augment=True, a small set of lightweight augmentations are added
    (rotation / small translation). These are safe for MNIST and help generalization.
    """
    if augment:
        return transforms.Compose(
            [
                # transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )


def test_transforms():
    """Return deterministic transforms for evaluation / test sets.

    Usually this is just ToTensor + Normalize. Kept as a helper so callers can
    explicitly use the same normalization as training without accidental augmentations.
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
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
        train_losses.append(loss)

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
