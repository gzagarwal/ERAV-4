# CIFAR-10 Image Classification

This project aims to train a deep learning model to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. The model developed here is based on a Convolutional Neural Network (CNN) architecture.

## Model Architecture

The model architecture is a Convolutional Neural Network designed for image classification. It consists of the following layers:

- **Conv2d-1:** Input: 3 channels, Output: 16 channels, Kernel Size: 3x3, Stride: 1, Padding: 1.
- **BatchNorm2d-2:** Normalizes the output of the first convolutional layer.
- **ReLU-3:** Applies the ReLU activation function.
- **Dropout-4:** Applies dropout with a rate of 0.05 to the output.
- **Conv2d-5:** Input: 16 channels, Output: 32 channels, Kernel Size: 3x3, Stride: 2 (Downsampling), Padding: 1.
- **BatchNorm2d-6:** Normalizes the output.
- **ReLU-7:** Applies ReLU activation.
- **Dropout-8:** Applies dropout with a rate of 0.05.
- **Conv2d-9:** Input: 32 channels, Output: 64 channels, Kernel Size: 3x3, Stride: 1, Padding: 1.
- **BatchNorm2d-10:** Normalizes the output.
- **ReLU-11:** Applies ReLU activation.
- **Dropout-12:** Applies dropout with a rate of 0.05.
- **Conv2d-13:** Input: 64 channels, Output: 64 channels, Kernel Size: 3x3, Stride: 1, Padding: 1.
- **BatchNorm2d-14:** Normalizes the output.
- **ReLU-15:** Applies ReLU activation.
- **Dropout-16:** Applies dropout with a rate of 0.05.
- **Conv2d-17:** Input: 64 channels, Output: 64 channels, Kernel Size: 3x3, Stride: 1, Padding: 1.
- **BatchNorm2d-18:** Normalizes the output.
- **ReLU-19:** Applies ReLU activation.
- **Conv2d-20:** Input: 64 channels, Output: 96 channels, Kernel Size: 3x3, Stride: 2 (Downsampling), Padding: 1.
- **BatchNorm2d-21:** Normalizes the output.
- **ReLU-22:** Applies ReLU activation.
- **Dropout-23:** Applies dropout with a rate of 0.05.
- **Conv2d-24:** Input: 96 channels, Output: 96 channels, Kernel Size: 3x3, Stride: 1, Padding: 0. This appears to be a depthwise separable convolution operation combined with the next pointwise convolution.
- **Conv2d-25:** Input: 96 channels, Output: 128 channels, Kernel Size: 1x1, Stride: 1, Padding: 0. This is likely the pointwise part of a depthwise separable convolution.
- **BatchNorm2d-26:** Normalizes the output.
- **ReLU-27:** Applies ReLU activation.
- **Dropout-28:** Applies dropout with a rate of 0.05.
- **DWSeparableConv-29:** Depthwise Separable Convolution (details from Conv2d-24 and Conv2d-25).
- **Conv2d-30:** Input: 128 channels, Output: 128 channels, Kernel Size: 3x3, Stride: 1, Padding: 0. Likely depthwise part of another depthwise separable convolution.
- **Conv2d-31:** Input: 128 channels, Output: 128 channels, Kernel Size: 1x1, Stride: 1, Padding: 0. Likely pointwise part of another depthwise separable convolution.
- **BatchNorm2d-32:** Normalizes the output.
- **ReLU-33:** Applies ReLU activation.
- **Dropout-34:** Applies dropout with a rate of 0.05.
- **DWSeparableConv-35:** Depthwise Separable Convolution (details from Conv2d-30 and Conv2d-31).
- **Conv2d-36:** Input: 128 channels, Output: 128 channels, Kernel Size: 3x3, Stride: 2 (Downsampling), Padding: 0. Likely depthwise part of a depthwise separable convolution with downsampling.
- **Conv2d-37:** Input: 128 channels, Output: 128 channels, Kernel Size: 1x1, Stride: 1, Padding: 0. Likely pointwise part of a depthwise separable convolution with downsampling.
- **BatchNorm2d-38:** Normalizes the output.
- **ReLU-39:** Applies ReLU activation.
- **Dropout-40:** Applies dropout with a rate of 0.05.
- **DWSeparableConv-41:** Depthwise Separable Convolution (details from Conv2d-36 and Conv2d-37).
- **Conv2d-42:** Input: 128 channels, Output: 128 channels, Kernel Size: 3x3, Stride: 1, Padding: 0. Likely depthwise part of a final depthwise separable convolution.
- **BatchNorm2d-43:** Normalizes the output.
- **ReLU-44:** Applies ReLU activation.
- **AdaptiveAvgPool2d-45:** Applies adaptive average pooling to reduce spatial dimensions to 1x1.
- **Conv2d-46:** Input: 128 channels, Output: 10 channels (for 10 classes), Kernel Size: 1x1, Stride: 1, Padding: 0. This acts as the final classification layer.

The network utilizes Batch Normalization and ReLU activation after most convolutional layers, along with Dropout for regularization. It also incorporates Depthwise Separable Convolutions for potentially more efficient processing. The spatial dimensions are reduced using strided convolutions and finally with an Adaptive Average Pooling layer before the final classification layer. The total number of trainable parameters is 154,544.

## Training and Test Results Analysis

The following plots show the training and test loss and accuracy over 50 epochs:

**Training Loss:**
The training loss decreases consistently and rapidly in the initial epochs, indicating that the model is learning effectively. After epoch 10, the rate of decrease slows down significantly, but the loss continues to drop gradually throughout the remaining epochs, suggesting the model is still improving its fit to the training data.

**Training Accuracy:**
The training accuracy increases rapidly in the initial epochs, mirroring the decrease in training loss. Similar to the loss, the rate of increase slows down after epoch 10, but the accuracy continues to climb steadily, reaching over 88% by the end of training. This indicates the model is becoming increasingly proficient at classifying the training data.

**Test Loss:**
The test loss also decreases in the initial epochs, following a similar trend to the training loss. After epoch 10, the test loss continues to decrease, but at a slower rate. The test loss remains relatively close to the training loss throughout the training process.

**Test Accuracy:**
The test accuracy increases in the initial epochs, tracking the training accuracy. After epoch 10, the test accuracy continues to increase, but with some minor fluctuations. The test accuracy is consistently slightly lower than the training accuracy, which is expected.

**Comparison and Conclusion:**
Comparing the training and test plots, there is no significant divergence between the training and test curves. Both the training and test loss decrease, and both the training and test accuracy increase over the 50 epochs. The test accuracy is consistently slightly lower than the training accuracy, which is normal and does not indicate significant overfitting. The model appears to be learning effectively and generalizes reasonably well to the unseen test data. The step size of the learning rate scheduler at epoch 10 had a clear positive impact on both training and test performance, leading to a steeper improvement curve after this point. The model achieves a respectable test accuracy of over 87% by the end of training.

