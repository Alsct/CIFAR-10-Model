Overview
--------

This project implements a custom Convolutional Neural Network (CNN) architecture for image classification on the CIFAR-10 dataset. The model features an innovative expert-gated architecture that dynamically weights multiple convolutional paths, allowing the network to adaptively select the most relevant features for each input.

Model Architecture
------------------

### Core Components

1.  **Stem Layer**

    -   Initial feature extraction with 3x3 convolution
    -   Batch normalization and ReLU activation
    -   Converts input images to low-level feature representations
2.  **Expert-Gated Blocks**

    -   **Expert Branch**: Uses global average pooling and fully connected layers to generate attention weights
    -   **Convolutional Layers**: K parallel convolutional paths with batch normalization and ReLU
    -   **Dynamic Weighting**: Expert branch produces softmax weights to combine outputs from K conv paths
3.  **Backbone**

    -   Stack of N expert-gated blocks (configurable, default: 8 blocks)
    -   Each block processes features and passes them to the next block
4.  **Classifier**

    -   Global average pooling to reduce spatial dimensions
    -   Fully connected layer for final classification (10 classes for CIFAR-10)

### Model Configuration

-   **Input Channels**: 3 (RGB)
-   **Output Channels**: 64
-   **Number of Blocks**: 8
-   **Expert Paths (K)**: 3
-   **Reduction Factor**: 8
-   **Classes**: 10 (CIFAR-10)

Key Features
------------

-   **Adaptive Feature Selection**: Expert branch dynamically weights different convolutional paths
-   **Residual-like Architecture**: Information flows through multiple blocks for deep feature learning
-   **Data Augmentation**: Random horizontal flip, random crop, and color jitter for improved generalization
-   **Advanced Optimization**: AdamW optimizer with cosine annealing learning rate scheduling

Training Details
----------------

-   **Dataset**: CIFAR-10 (50,000 training, 10,000 test images)
-   **Batch Size**: 64
-   **Epochs**: 100
-   **Learning Rate**: 0.0003
-   **Weight Decay**: 0.001
-   **Optimizer**: AdamW
-   **Loss Function**: CrossEntropyLoss
-   **Scheduler**: CosineAnnealingLR

Data Preprocessing
------------------

-   **Normalization**: CIFAR-10 specific mean and std values
-   **Training Augmentations**:
    -   Random horizontal flip
    -   Random crop with padding
    -   Color jitter (brightness, contrast, saturation)
-   **Test Preprocessing**: Standard normalization only

Performance Monitoring
----------------------

The model tracks and visualizes:

-   Training and validation accuracy over epochs
-   Training and validation loss over epochs
-   Real-time epoch timing and performance metrics

Technical Implementation
------------------------

-   Built with PyTorch framework
-   Modular design with separate classes for each component
-   GPU acceleration support (CUDA when available)
-   Comprehensive logging and visualization

Architecture Innovation
-----------------------

The expert-gated mechanism allows the model to:

-   Learn which convolutional features are most relevant for each input
-   Dynamically adjust feature importance through learned attention weights
-   Achieve better feature representation through adaptive path selection
