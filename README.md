# Re-Creating Micrograd

A minimal neural network and autograd engine implemented from scratch in Python, inspired by micrograd. This repository provides a simple framework for building and training multi-layer perceptrons (MLPs) with automatic differentiation.

## Features

- **Autograd Engine:** Scalar-based reverse-mode automatic differentiation (`Value` class in `Engine.py`).
- **Neural Network Components:** Neuron, Layer, and MLP classes (`nn.py`).
- **Loss Functions:** Supports SSE, MSE, and cross-entropy (`utils.py`).
- **Activation Functions:** tanh, sigmoid, relu, leaky relu.
- **Softmax:** For multi-class outputs (`utils.py`).

## File Structure

- `Engine.py`: Core autograd engine with the `Value` class.
- `nn.py`: Neural network components (Neuron, Layer, MLP).
- `utils.py`: Utility functions for loss calculation and softmax.

## Example Usage

```python
from nn import MLP
from utils import loss, softMax
import numpy as np

# Create a network with 3 inputs, one hidden layer of 4, and 2 outputs
net = MLP([3, 4, 2])

# Example input
x = np.array([1.0, 2.0, 3.0])

# Forward pass
output = net(x)

# Example target
y_true = [1.0, 0.0]

# Compute loss
l = loss(output, y_true, criterion='mse')

# Backward pass
l.backward()
```

## Requirements

- Python 3.7+
- numpy

## How It Works

- **Autograd:** The `Value` class tracks operations and computes gradients via `.backward()`.
- **Neural Network:** `MLP` builds a feedforward network from layers of `Neuron`.
- **Loss & Softmax:** `utils.py` provides loss functions and softmax for output normalization.


---
