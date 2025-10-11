from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn

# Metric collectors (populated during a run)
train_losses = []
train_acc = []

# Add memory management utilities
import gc


def train_transforms(augment: bool = False):
    """Returns data transforms for CIFAR-100 training.

    Args:
        augment: If True, applies strong augmentation strategy:
                - Random crop with padding
                - Random horizontal flip
                - Random erasing (cutout)
                - Normalization with CIFAR-100 mean/std
                
    Returns:
        torchvision.transforms.Compose with appropriate transforms
    """
    CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100_STD = [0.2675, 0.2565, 0.2761]
    if augment:
        return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        # Random Erasing (Cutout) is critical for CIFAR performance 
        # as a form of strong regularization
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])


def test_transforms():
    """Returns deterministic transforms for CIFAR-100 evaluation/test sets.
    
    Includes:
        - ToTensor conversion
        - Normalization with CIFAR-100 specific mean/std values
        
    No augmentations are applied to ensure consistent evaluation.
    """
    CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100_STD = [0.2675, 0.2565, 0.2761]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)      
])


def train(model, device, train_loader, optimizer, epoch):
    """Trains the model for one epoch.
    
    Features:
        - Label smoothing (0.1) for better generalization
        - Gradient clipping to prevent exploding gradients
        - Progress bar with live loss and accuracy metrics
        - Memory optimization with scalar metric storage
    """
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    epoch_losses = []
    epoch_accs = []
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        # Store loss as scalar value instead of tensor to save memory
        epoch_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Store accuracy as scalar
        current_acc = 100*correct/processed
        epoch_accs.append(current_acc)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={current_acc:0.2f}')
        
        # Clear intermediate tensors
        del y_pred, pred, loss
        
        # Periodic memory cleanup
        if batch_idx % 50 == 0:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    # Store only epoch averages instead of all batch values
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    avg_epoch_acc = sum(epoch_accs) / len(epoch_accs)
    
    train_losses.append(avg_epoch_loss)
    train_acc.append(avg_epoch_acc)
    
    # Clear epoch data
    del epoch_losses, epoch_accs
    
    # Final memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


