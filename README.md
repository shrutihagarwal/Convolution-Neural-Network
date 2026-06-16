# 🧩 CNN Architectures From the Ground Up

A from-scratch implementation of the landmark Convolutional Neural Network architectures that defined the evolution of deep learning for computer vision — built layer-by-layer in Keras/TensorFlow to understand *why* each architecture innovated the way it did, not just how to call a pretrained model.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architectures Implemented](#architectures-implemented)
- [Implementation Highlights](#implementation-highlights)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Architecture Evolution & Key Ideas](#architecture-evolution--key-ideas)
- [Key Learnings](#key-learnings)

---

## Overview

Rather than only using pretrained CNNs as black boxes, this project reconstructs the major architectures from the deep learning literature directly in code — from the original LeNet-5 (1998) through to modern designs like MobileNet's depthwise-separable convolutions. Each implementation is built to mirror the architecture described in its original paper, including non-obvious details like GoogLeNet's auxiliary classifiers and ResNet's identity/convolutional skip-connection blocks.

---

## Architectures Implemented

| Architecture | Year | Key Innovation | Implemented Here |
|--------------|------|----------------|-------------------|
| **LeNet-5** | 1998 | First successful CNN (digit recognition) | 2 variants — sigmoid (original) and ReLU (modernized) |
| **CNN (CIFAR-10 / MNIST baseline)** | — | Standard Conv-Pool-Dense baseline | Keras Sequential models |
| **AlexNet** | 2012 | Deep CNN + ReLU + Dropout at ImageNet scale | 2 variants (binary and 1000-class heads) |
| **VGG16 / VGG19** | 2014 | Deep, uniform 3×3 conv stacks | Multiple variants incl. binary classification head |
| **ResNet (18/50-style)** | 2015 | Residual / skip connections to enable very deep networks | Custom residual blocks + full `ResNet50()` with identity & convolutional blocks |
| **GoogLeNet / Inception v1** | 2014 | Multi-scale "Inception block" + auxiliary classifiers | Full `Inception_block()` + `GoogLeNet()` with 9 inception blocks and 2 auxiliary heads |
| **MobileNet** | 2017 | Depthwise-separable convolutions for efficiency | Custom `convolution_block()` using `DepthwiseConv2D` |
| **Transfer Learning (VGG16)** | — | Reusing ImageNet-pretrained features | Frozen/unfrozen layer fine-tuning + intermediate-layer feature extraction |

---

## Implementation Highlights

### Residual Blocks (ResNet)
```python
def identity_block(X, f, filters, stage, block):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1,1))(X); X = BatchNormalization()(X); X = Activation('relu')(X)
    X = Conv2D(F2, (f,f), padding='same')(X); X = BatchNormalization()(X); X = Activation('relu')(X)
    X = Conv2D(F3, (1,1))(X); X = BatchNormalization()(X)
    X = Add()([X, X_shortcut])   # <-- the skip connection that defines ResNet
    return Activation('relu')(X)
```
This mirrors the bottleneck identity block from the original ResNet paper (He et al., 2015) — the `Add()` skip connection is what allows gradients to flow through very deep networks without vanishing.

### Inception Block (GoogLeNet)
```python
def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    path1 = Conv2D(f1, (1,1), activation='relu')(input_layer)
    path2 = Conv2D(f2_conv1, (1,1))(input_layer); path2 = Conv2D(f2_conv3, (3,3))(path2)
    path3 = Conv2D(f3_conv1, (1,1))(input_layer); path3 = Conv2D(f3_conv5, (5,5))(path3)
    path4 = MaxPooling2D((3,3), strides=1, padding='same')(input_layer)
    path4 = Conv2D(f4, (1,1))(path4)
    return concatenate([path1, path2, path3, path4], axis=-1)
```
Four parallel paths process the input at different receptive field sizes (1×1, 3×3, 5×5, and pooling) and concatenate the results — the core idea that lets the network "choose" the right scale rather than committing to one filter size per layer.

### Depthwise Separable Convolutions (MobileNet)
```python
def convolution_block(input_layer, strides, filters):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x); x = ReLU()(x)
    x = Conv2D(filters, kernel_size=1, strides=1)(x)   # pointwise convolution
    return ReLU()(BatchNormalization()(x))
```
Splitting a standard convolution into a depthwise (spatial) pass and a pointwise (channel-mixing) pass drastically reduces parameter count — the key trick that makes MobileNet viable on mobile/edge devices.

---

## Project Structure

```
Convolution-Neural-Network/
├── CNN.py          # All 9 architecture topics implemented sequentially
└── README.md
```

The script is organized by "Topic" markers covering, in order: LeNet → CNN baseline → AlexNet → ResNet → VGG16 → VGG19 → MobileNet → Transfer Learning → GoogLeNet.

---

## Technologies Used

| Library | Usage |
|---------|-------|
| `TensorFlow` / `Keras` | All model construction (Sequential and Functional API) |
| `keras.applications` | Pretrained VGG16 for transfer learning comparison |
| `PyTorch` (`torch`, `torch.nn`) | Imported for architecture cross-referencing |
| `numpy`, `matplotlib` | Supporting data handling and visualization |

---

## How to Run

```bash
pip install tensorflow keras torch matplotlib numpy
python CNN.py
```

Each architecture section is self-contained — `model.summary()` is called after each model definition to inspect parameter counts and layer shapes. This is best run cell-by-cell in Jupyter (the file retains `# In[n]:` cell markers from its notebook origin) to compare architectures side by side.

---

## Architecture Evolution & Key Ideas

This implementation effectively traces the central tension in CNN design history: **how to go deeper without the network breaking.**

- **LeNet → AlexNet**: scale (more filters, more layers, ReLU instead of sigmoid/tanh, dropout for regularization at ImageNet scale).
- **AlexNet → VGG**: simplicity and uniformity (only 3×3 convolutions, stacked deeper) showed depth alone improves accuracy.
- **VGG → GoogLeNet**: efficiency via multi-scale processing instead of just stacking — fewer parameters than VGG despite comparable depth, plus auxiliary classifiers to combat vanishing gradients in early training.
- **→ ResNet**: directly solves the vanishing-gradient/degradation problem with identity skip connections, enabling networks with 50, 101, even 152+ layers.
- **→ MobileNet**: shifts the goal from "more accurate at any cost" to "accurate enough at minimal compute" via depthwise-separable convolutions, for deployment on resource-constrained devices.

---

## Key Learnings

1. **Skip connections are not a minor detail** — implementing ResNet's identity block by hand makes clear why removing the `Add()` step would make very deep networks untrainable.
2. **Multi-scale processing (Inception) vs. uniform depth (VGG)** represent two genuinely different philosophies for the same goal, each with real trade-offs in parameter count and accuracy.
3. **Transfer learning is not just "reuse a model"** — selectively freezing layers (`layer.trainable = False`) and extracting intermediate activations (e.g., `block4_pool`) shows how pretrained features generalize at different depths.
4. **Efficiency-oriented design (MobileNet) requires architectural creativity**, not just smaller versions of existing networks — depthwise separable convolutions are a structurally different operation, not a scaled-down `Conv2D`.

---

## References

- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition.
- Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). ImageNet Classification with Deep CNNs. *NeurIPS*.
- Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG). *arXiv:1409.1556*.
- Szegedy, C. et al. (2014). Going Deeper with Convolutions (GoogLeNet). *CVPR*.
- He, K. et al. (2015). Deep Residual Learning for Image Recognition (ResNet). *CVPR*.
- Howard, A. et al. (2017). MobileNets: Efficient CNNs for Mobile Vision Applications. *arXiv:1704.04861*.

---

*A architecture-by-architecture study of CNN design history, built to internalize the reasoning behind each innovation rather than treat models as black boxes.*
