# CIFAR-100 Classification with ResNet34

A PyTorch implementation of a modified ResNet34 architecture for CIFAR-100 image classification, optimized for high accuracy and efficient training.

## Project Overview

This project implements a customized ResNet34 architecture for classifying images from the CIFAR-100 dataset. The implementation focuses on achieving high accuracy (target: 75%) through architectural modifications and modern training techniques.

## Architecture Details

### ResNet34 Architecture
- Input: 32x32x3 RGB images
- Initial Layer: 3x3 conv, 64 filters (adapted for CIFAR-100's smaller input size)
- Residual Block Configuration: [3, 4, 6, 3]
- No initial MaxPool layer (adapted for smaller input size)
- Downsampling through strided convolutions
- Dropout (p=0.2) before final FC layer
- Output: 100 classes


## Implementation Details

### Training Configuration
- Epochs: 250
- Batch Size: 128
- Base Learning Rate: 0.1
- Momentum: 0.9
- Weight Decay: 0.0005
- Optimizer: SGD with Nesterov momentum
- Loss Function: CrossEntropyLoss with label smoothing (0.1)

### Advanced Features
1. **Learning Rate Schedule**
   - OneCycleLR policy
   - 10% warmup period
   - Initial LR = max_lr/10
   - Final LR = max_lr/100

2. **Data Augmentation Pipeline**
   - Random crop (32x32) with padding=4
   - Random horizontal flip
   - Random erasing (cutout) with p=0.5
   - Normalization with CIFAR-100 mean/std values

3. **Training Optimizations**
   - Gradient clipping (max_norm=1.0)
   - Memory management with periodic cache clearing
   - Pin memory for faster data transfer
   - Multi-worker data loading

### Memory Optimizations
- Periodic GPU cache clearing
- Gradient clipping to prevent exploding gradients
- Efficient tensor operations
- Proper cleanup of intermediate tensors

## Model Performance

Training metrics from the implementation:
- Training Loss: Monitored and plotted
- Training Accuracy: Tracked per epoch
- Test Loss: Evaluated after each epoch
- Test Accuracy: Monitored for model selection

### Checkpointing
- Regular checkpoints saved during training
- Best model saved based on validation accuracy
- Support for training resumption