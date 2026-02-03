# CNN Assignment: Convolutional Neural Networks

## Problem Description
This project explores the role of Convolutional Neural Networks (CNNs) in image classification. We interpret neural networks not as black boxes but as architectural components with specific inductive biases. The goal is to compare a standard Fully Connected Network (Baseline) against a custom CNN and analyze how architectural choices like Kernel Size affect performance.

## Dataset Description
**Dataset**: CIFAR-10
- **Source**: `tensorflow.keras.datasets.cifar10`
- **Content**: 60,000 color images (32x32 pixels) in 10 classes.
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Split**: 50,000 Training samples, 10,000 Testing samples.
- **Preprocessing**: Normalized pixel values to the range [0, 1].


## Architectures

### 1. Baseline Model (MLP)
A simple feed-forward network to establish a performance floor.
- **Input**: Flattened 32x32x3 image (3072 input features).
- **Hidden Layers**: Two Dense layers (512 and 256 units) with ReLU activation.
- **Output**: 10 units with Softmax activation.
- **Parameters**: ~1.7 million (High parameter count due to dense connections).

### 2. Custom CNN Model
A standard CNN design for feature extraction.
- **Layers**:
    - Conv2D (32 filters, 3x3) + ReLU + MaxPool(2x2)
    - Conv2D (64 filters, 3x3) + ReLU + MaxPool(2x2)
    - Conv2D (64 filters, 3x3) + ReLU
    - Flatten
    - Dense (64) + ReLU
    - Output (10)
- **Parameters**: Significantly fewer than the MLP (~100k-200k), yet more effective.