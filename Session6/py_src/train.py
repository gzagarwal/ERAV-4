from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms

# Metric collectors (populated during a run)
train_losses = []
train_acc = []


def train_transforms(augment: bool = False):
    """Return a torchvision.transforms.Compose for training.

    If augment=True, a small set of lightweight augmentations are added
    (rotation / small translation). These are safe for MNIST and help generalization.
    """
    if augment:
        return transforms.Compose([
            transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def test_transforms():
    """Return deterministic transforms for evaluation / test sets.

    Usually this is just ToTensor + Normalize. Kept as a helper so callers can
    explicitly use the same normalization as training without accidental augmentations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


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
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
