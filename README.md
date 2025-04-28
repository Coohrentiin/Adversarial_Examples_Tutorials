# Adversarial Examples for Image Recognition

This repository contains a tutorial on creating adversarial examples to fool deep learning image classifiers. The goal is to demonstrate how adding carefully crafted perturbations (that are often imperceptible to humans) can cause well-trained models to misclassify images.

You can find some context in my [Medium post here](https://medium.com/@corentin.soubeiran/adversarial-attacks-when-ai-gets-bamboozled-by-pixel-pranks-also-for-newbies-566f8f97094d)

If you like it don't forget to "⭐" this repo ;) 
## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Tutorial](#tutorial)
  - [Understanding Adversarial Examples](#understanding-adversarial-examples)
  - [Fast Gradient Sign Method (FGSM)](#fast-gradient-sign-method)
  - [Targeted vs. Untargeted Attacks](#targeted-vs-untargeted-attacks)
- [Implementations](#implementations)
- [Contributing](#contributing)

## Introduction

Adversarial examples are inputs to machine learning models designed to cause the model to make a mistake. In the context of image classification, these are images with small, often imperceptible perturbations that cause a model to misclassify the image.

This tutorial focuses on generating adversarial examples for pre-trained convolutional neural networks (CNNs) like ResNet, VGG, and MobileNet. You'll learn how to:

1. Generate untargeted adversarial examples (causing any misclassification)
2. Create targeted adversarial examples (forcing classification into a specific class)
3. Visualize and understand the perturbations
4. Evaluate the robustness of different models against adversarial attacks

## Repository Structure
```
.
├── README.md                  # Main documentation
├── data/                      # Data directory 
│   └── sample_images/         # Sample images for testing
│       ├── cat.jpg
│       ├── dog.jpg
│       └── ...
├── examples/                  # Example implementations
│   ├── fgsm_example.py        # FGSM attack example
│   ├── pgd_example.py         # PGD attack example
│   └── targeted_attack.py     # Targeted attack example
├── notebooks/                 # Jupyter notebooks
│   ├── Attacks_Introduction_notebook.ipynb  # Introduction to adversarial examples
│   └── Attacks_Deepening_notebook.ipynb     # Some additionnal insights
├── requirements.txt           # Required packages
└── src/                       # Source code
  ├── __init__.py            # Package initialization
  ├── adversarial_attackes   # 4 classical adversarial attacks examples
  ├── models.py              # To load classical models
  ├── utils/                 # Utility functions
  └── visualization.py   # Result visualization tools

```

## Installation

To get started with this tutorial, clone the repository and install the required dependencies:

```bash
git clone https://github.com/coohrentiin/adversarial-examples.git
cd adversarial-examples
pip install -r requirements.txt
```

### Requirements

The main dependencies are:
- numpy>=1.19.0
- torch>=1.7.0
- torchvision>=0.8.0
- matplotlib>=3.3.0
- jupyter>=1.0.0
- pillow>=8.0.0
- scikit-learn>=0.24.0
- tqdm>=4.50.0
- ipykernel>=5.3.0
- lpips (https://pypi.org/project/lpips/)

## Usage Examples

Here's a quick example of how to generate an adversarial example using the Fast Gradient Sign Method:

```python
from src.models import load_pretrained_model
from src.attacks import fgsm_attack
from src.utils import load_image, preprocess_image, show_image
import torch

# Load a pretrained model (ResNet-50)
model = load_pretrained_model('resnet50')

# Load and preprocess an image
image_path = 'data/sample_images/cat.jpg'
original_image = load_image(image_path)
input_tensor = preprocess_image(original_image)

# Generate an adversarial example (untargeted)
epsilon = 0.01  # Perturbation magnitude
adversarial_tensor = fgsm_attack(model, input_tensor, target=None, epsilon=epsilon)

# Display the original and adversarial images
show_image(original_image, "Original Image")
adversarial_image = deprocess_tensor(adversarial_tensor)
show_image(adversarial_image, "Adversarial Image")

# Compare predictions
original_pred = predict(model, input_tensor)
adversarial_pred = predict(model, adversarial_tensor)
print(f"Original prediction: {original_pred}")
print(f"Adversarial prediction: {adversarial_pred}")
```

## Tutorial

### Understanding Adversarial Examples

Adversarial examples were first introduced by Szegedy et al. in 2013, who discovered that neural networks can be fooled by adding small perturbations to the input that are often imperceptible to humans but cause the network to misclassify the input.

These perturbations are not random noise but are specifically optimized to maximize the loss function of the neural network, pushing it towards incorrect predictions.

### Fast Gradient Sign Method (FGSM)

The Fast Gradient Sign Method (FGSM) is one of the simplest and most widely used methods for generating adversarial examples. Introduced by Goodfellow et al. in 2014, FGSM works by using the gradients of the neural network to create an adversarial example.

The method computes the gradient of the loss with respect to the input image, then takes a small step in the direction that maximizes the loss:

```
x_adv = x + ε * sign(∇_x J(θ, x, y))
```

Where:
- x_adv is the adversarial example
- x is the original input image
- ε is a small scalar value that controls the magnitude of the perturbation
- ∇_x J(θ, x, y) is the gradient of the loss function with respect to the input x
- sign() is the sign function that returns -1, 0, or 1 depending on the sign of the input

### Targeted vs. Untargeted Attacks

Adversarial attacks can be broadly categorized into two types:

1. **Untargeted Attacks**: The goal is to cause any misclassification, regardless of which incorrect class the model predicts.

2. **Targeted Attacks**: The goal is to force the model to predict a specific target class.

In targeted attacks, instead of maximizing the loss for the correct class, we minimize the loss for the target class. This typically requires more computational effort but can demonstrate more dramatic results (like making a cat image be classified as a specific class like "dog" or "airplane").

### Project Goals

By the end of this tutorial, students should be able to:

1. Understand the concept of adversarial examples and why they pose security concerns
2. Implement different adversarial attack methods (FGSM, PGD, etc.)
3. Create adversarial examples that cause specific misclassifications
4. Evaluate the vulnerability of different models to adversarial attacks
5. Develop intuition about the robustness of neural networks

## Implementations

The repository includes implementations of several adversarial attack methods:

1. **Fast Gradient Sign Method (FGSM)** - A simple one-step method that uses the sign of the gradient.

2. **Projected Gradient Descent (PGD)** - An iterative version of FGSM with small steps.

3. **DeepFool** - A method that computes the minimum perturbation to cross the decision boundary.

4. **Carlini & Wagner (C&W)** - A more sophisticated optimization-based attack.

Each implementation includes detailed documentation and examples in the corresponding notebooks.

## Contributing

Contributions to improve the tutorial are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
