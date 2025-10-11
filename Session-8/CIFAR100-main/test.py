import torch
import torch.nn.functional as F
import gc

test_losses = []
test_acc = []


def test(model, device, test_loader):
    """Evaluates the model on test data.
    
    Features:
        - No gradient computation for memory efficiency
        - Periodic memory cleanup every 50 batches
        - Cross-entropy loss calculation
        - Maintains running test metrics (loss and accuracy)
        - GPU memory optimization with explicit cache clearing
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Clear intermediate tensors
            del output, pred
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    test_losses.append(test_loss)
    test_acc.append(test_accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    
    # Final memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
