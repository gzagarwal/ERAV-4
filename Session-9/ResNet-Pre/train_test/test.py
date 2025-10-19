import torch.nn as nn
import torch.nn.functional as F
import torch

test_losses = []
test_acc = []

criterion = nn.CrossEntropyLoss()


def test(model, device, test_loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    n_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = (
                data.to(device),
                data.new_tensor(target, dtype=torch.long).to(device)
                if isinstance(target, torch.Tensor)
                else target.to(device),
            )  # safe cast if needed
            logits = model(data)
            # batch loss (mean)
            loss = criterion(logits, target)  # CrossEntropyLoss expects raw logits
            bs = data.size(0)

            loss_sum += loss.item() * bs  # weight by batch size
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            n_samples += bs

    avg_loss = loss_sum / n_samples
    avg_acc = 100.0 * correct / n_samples

    test_losses.append(avg_loss)
    test_acc.append(avg_acc)

    print(
        f"\nTest set: Average loss: {avg_loss:.4f}, "
        f"Accuracy: {correct}/{n_samples} ({avg_acc:.2f}%)\n"
    )

    return avg_loss, avg_acc
