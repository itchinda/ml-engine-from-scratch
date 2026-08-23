# ml_framework/__init__.py

# Importe les classes depuis tes sous-modules pour les rendre directement accessibles
from .layers import LinearLayer, ActivationLayer, Conv2D, MaxPool2D, Flatten, Dropout
from .activations import ReLU, ReLUDerivative, Sigmoid, SigmoidDerivative
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss
from .optimizers import Adam, SGD
from .models import Sequential

__all__ = [
    "LinearLayer",
    "ActivationLayer",
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "Dropout",
    "ReLU",
    "ReLUDerivative",
    "Sigmoid",
    "SigmoidDerivative",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "Adam",
    "SGD",
    "Sequential"
]