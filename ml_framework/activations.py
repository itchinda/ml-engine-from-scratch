import numpy as np

# =========================================================================
# ======================== ACTIVATION FUNCTIONS ===========================
# =========================================================================

def Tanh(X: np.ndarray) -> np.ndarray:
    """Hyperbolic Tangent activation function."""
    return np.tanh(X)

def TanhDerivative(X: np.ndarray) -> np.ndarray:
    """Derivative of the Hyperbolic Tangent function."""
    return 1 - np.tanh(X)**2

def ReLU(X: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function."""
    return np.maximum(0, X)

def ReLUDerivative(X: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU function. Returns boolean mask castable during multiplication."""
    return (X > 0).astype(float)

def Sigmoid(X: np.ndarray) -> np.ndarray:
    """Sigmoid activation function. Squashes values between 0 and 1."""
    # Clip values to avoid overflow in np.exp()
    X_clipped = np.clip(X, -500, 500)
    return 1.0 / (1.0 + np.exp(-X_clipped))

def SigmoidDerivative(X: np.ndarray) -> np.ndarray:
    """Derivative of the Sigmoid function."""
    s = Sigmoid(X)
    return s * (1.0 - s)

def Softmax(logits: np.ndarray) -> np.ndarray:
    # Subtracting the maximum value for numerical stability (prevents overflow)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted_logits)
    return exps / np.sum(exps, axis=1, keepdims=True)