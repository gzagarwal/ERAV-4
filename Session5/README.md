# MNIST Model Training Experiments - Session 5

## Overview
This repository contains four iterations of experiments training neural network models on the MNIST dataset. The goal was to achieve high accuracy (>95%) with models having relatively few parameters. Each notebook documents a different approach or improvement, with results and model details.

## Notebooks
- **Iteration1Session5.ipynb**
- **Iteration2_99_34_80K.ipynb**
- **Iteration3_99_20_19K.ipynb**
- **Iteration4_99_42_19K.ipynb**

## Description
Each notebook follows a similar structure:
- Data loading and augmentation using torchvision
- Model definition (PyTorch)
- Training and evaluation loops
- Accuracy and loss tracking
- Model summary and results

## Model Architecture Comparison Table

| Notebook                | Test Accuracy | BatchNorm Layers         | # Parameters | Fully Connected Layers | Activation Functions         |
|-------------------------|--------------|-------------------------|--------------|-----------------------|------------------------------|
| Iteration1Session5      | ~99.25%      | None                    | 24,245       | 2 (65, 10)            | ReLU, LogSoftmax             |
| Iteration2_99_34_80K    | ~99.34%      | 2x BatchNorm2d          | 80,501       | 2 (65, 10)            | ReLU, LogSoftmax             |
| Iteration3_99_20_19K    | ~99.20%      | 2x BatchNorm2d, 1x1d    | 14,354       | 2 (32, 10)            | ReLU, LogSoftmax             |
| Iteration4_99_42_19K    | ~99.42%      | 2x BatchNorm2d, 1x1d    | 19,354       | 2 (48, 10)            | ReLU, LogSoftmax             |

**Key Points:**
- All models use ReLU for hidden layers and LogSoftmax for output.
- Batch normalization is used in Iteration2, 3, and 4 (Iteration2: only 2d; Iteration3/4: 2d and 1d).
- Iteration3 and 4 achieve high accuracy with fewer parameters.

## Models & Results

### Iteration 1

### Iteration 2

### Iteration 3

### Iteration 4


## Requirements
- Python 3.x
- torch, torchvision, torchsummary
- tqdm, matplotlib (for plots)

## Acknowledgements
- MNIST dataset (Yann LeCun et al.)

For more details, see the individual notebooks.
