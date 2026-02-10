# Convolutional Neural Networks Assignment

## Context and Motivation
In this course, neural networks are not treated as black boxes but as architectural components whose design choices affect performance, scalability, and interpretability. This assignment focuses on convolutional layers as a concrete example of how inductive bias is introduced into learning systems.

Rather than following a recipe, we selected, analyzed, and experimented with a convolutional architecture using a real dataset.

## 1. Dataset Selection and Exploration (EDA)
**Selected Dataset**: **CIFAR-10**

### Justification
CIFAR-10 is a widely recognized benchmark for image classification. It is appropriate for this assignment because:
- **Image-based**: It consists of 3-channel RGB images, perfect for testing convolutional filters.
- **Complexity**: It is complex enough that simple MLPs struggle (providing a good baseline comparison) but small enough to train on a standard environment.
- **Multi-class**: 10 distinct classes require the model to learn discriminative features.

### EDA Analysis
- **Dataset Size**: 60,000 Total Images.
  - **Training**: 50,000 images.
  - **Testing**: 10,000 images.
- **Dimensions**: 32x32 pixels with 3 color channels (RGB).
- **Class Distribution**: Balanced (6,000 images per class).
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Preprocessing**: 
  - Pixel values were normalized to the range `[0, 1]` by dividing by 255.0.
  - No resizing was needed as 32x32 is the native resolution.

## 2. Baseline Model (Non-Convolutional)
To establish a reference point, we implemented a standard Multi-Layer Perceptron (MLP).

### Architecture
- **Input**: Flattened 32x32x3 image (3072 input features).
- **Hidden Layer 1**: Dense (512 units, ReLU activation).
- **Hidden Layer 2**: Dense (256 units, ReLU activation).
- **Output Layer**: Dense (10 units, Softmax activation).

### Observations
- **Parameters**: Approximately **1.7 Million**. The high number of parameters is due to the dense connections in the first layer ($3072 \times 512$ weights).
- **Limitations**: The model ignores the spatial structure of the image (treating pixels as independent features). It is prone to overfitting and computationally expensive relative to its performance.

## 3. Convolutional Architecture Design
We designed a custom CNN from scratch to leverage spatial hierarchies.

### Architecture Definition
1. **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation.
   - *Followed by*: MaxPooling2D (2x2).
2. **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation.
   - *Followed by*: MaxPooling2D (2x2).
3. **Conv2D Layer 3**: 64 filters, 3x3 kernel, ReLU activation.
4. **Classifier Head**:
   - Flatten.
   - Dense (64 units, ReLU).
   - Output (10 units, Softmax).

### Justification of Choices
- **Kernels (3x3)**: Small kernels are efficient and capable of capturing local details (edges, textures). Stacking them allows the network to learn larger patterns.
- **Pooling (2x2)**: Used to downsample feature maps, reducing the computational load and providing translation invariance (small shifts in the image don't change the extracted features).
- **Depth**: Increasing filters (32 $\rightarrow$ 64) allows the network to capture more complex combinations of features as it goes deeper.

## 4. Controlled Experiments (Baseline vs CNN)
We compared the Baseline MLP against the Custom CNN.

### Results
- **Parameter Efficiency**: The CNN has significantly fewer parameters (~100k-200k) compared to the Baseline (~1.7M), yet typically achieves higher accuracy.
- **Performance**: The CNN consistently outperforms the MLP on the validation set because it can generalize spatial patterns, whereas the MLP memorizes pixel locations.

## 5. Interpretation and Architectural Reasoning
### Why did Convolutional Layers outperform the Baseline?
Convolutional layers excelled because they align with the structure of image data. The MLP treats the pixel at (0,0) and (0,1) with no more relationship than (0,0) and (31,31). The CNN, through its kernels, explicitly looks for local relationships and patterns.

### Inductive Bias of Convolution
- **Locality**: Features in images are spatially localized (e.g., an eye is made of pixels close to each other).
- **Translation Equivariance**: A pattern (like a wheel) appearing in the top-left corner is the same feature as one in the bottom-right. Convolutional filters share weights across the entire image, allowing the model to detect the same feature anywhere.

### When is Convolution NOT Appropriate?
Convolution is not suitable for data where "neighboring" features have no inherent relationship, such as tabular data (e.g., a spreadsheet where column A is Age and column B is Income). Reordering columns in tabular data changes nothing about the data's meaning, but reordering pixels in an image destroys it.

## 6. Deployment in SageMaker
> **Important Note**: Training and deployment to an AWS SageMaker endpoint was **not possible** for this assignment due to account permission restrictions. The model execution and verification have been performed locally within the Jupyter Notebook.