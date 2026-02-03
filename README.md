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
