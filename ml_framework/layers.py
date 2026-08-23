import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from collections.abc import Callable
import pickle

# ==============================================================================
# =============== ABSTRACT BASE CLASS FOR NEURAL NETWORK LAYERS ================
# ==============================================================================
class NNLayer(ABC):
    """Abstract base class for all neural network layers."""
    
    def __init__(self):
        # Cache to store inputs for backpropagation
        self.input_data: Optional[np.ndarray] = None 

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Processes input data and returns the Neural Network layer's output."""
        pass

    @abstractmethod
    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Applies the chain rule.
        Takes the gradient from the next neural network layer, updates internal gradients, 
        and returns the gradient for the previous layer.
        """
        pass

    def train(self):
        """Sets the neural network layer to training mode."""
        self.training = True
        if hasattr(self, 'layers'):
            for layer in self.layers:
                layer.train()

    def eval(self):
        """Sets the neural network layer to evaluation (testing) mode."""
        self.training = False
        if hasattr(self, 'layers'):
            for layer in self.layers:
                layer.eval()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Allows the neural network layer to be called like a function (e.g., model(X))."""
        return self.forward(X)

    def get_parameters(self) -> dict:
        """Returns a dictionary of the NN layer's learnable parameters."""
        return {} # Default: no parameters (for Activation/Dropout)

    def set_parameters(self, params: dict):
        """Loads learnable parameters from a dictionary."""
        pass

    def save(self, filepath: str):
        """Extracts all parameters and saves them to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
        print(f"Model saved successfully to {filepath}")

    def load(self, filepath: str):
        """Loads parameters from a file and injects them into the model."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_parameters(params)
        print(f"Model loaded successfully from {filepath}")

    def extra_repr(self) -> str:
        """Allows subclasses to add custom string information (like in_features)."""
        return ""

    def __repr__(self) -> str:
        """The PyTorch-style recursive string representation."""
        # Get the class name (e.g., 'LinearLayer' or 'Sequential')
        name = self.__class__.__name__
        
        # Get specific layer info (e.g., 'in_features=1, out_features=32')
        extra_info = self.extra_repr()
        
        # Look for sub-layers (like inside Sequential)
        child_lines = []
        if hasattr(self, 'layers'):
            for i, child_layer in enumerate(self.layers):
                # Recursively get the string of the child
                child_str = repr(child_layer)
                indented = "\n".join([f"\t{line}" for line in child_str.split("\n")])
                child_lines.append(f"\t({i}): {indented.strip()}")
        
        # If it has no children, return the simple one-liner
        if not child_lines:
            return f"{name}({extra_info})"
            
        # If it has children, build the nested tree block
        newline = '\n' if extra_info else ''
        return f"{name}(\n{extra_info}{newline}" + "\n".join(child_lines) + "\n)"

    def __str__(self) -> str:
        return repr(self)


# ==============================================================================
# =============================== LINEAR NN LAYER ==============================
# ==============================================================================
class LinearLayer(NNLayer):
    """A fully connected (dense) linear layer."""
    
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator = np.random.default_rng()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # He/Xavier initialization for stable gradients
        self.weights = rng.standard_normal(size=(in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(shape=(1, out_features))
        
        # Placeholders for parameter gradients
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_data = X
        return X.dot(self.weights) + self.bias
    
    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        # Calculate gradients for parameters
        self.grad_weights = self.input_data.T.dot(upstream_gradient)
        self.grad_bias = np.sum(upstream_gradient, axis=0, keepdims=True)
        
        # Calculate gradient to pass backward to the previous layer
        return upstream_gradient.dot(self.weights.T)

    def get_parameters(self) -> dict:
        # Use .copy() to ensure we don't save references to active memory
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def set_parameters(self, params: dict):
        self.weights = params['weights']
        self.bias = params['bias']

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=True"

# ==============================================================================
# =============================== ACTIVATION NN LAYER ==============================
# ==============================================================================
class ActivationLayer(NNLayer):
    """Applies a non-linear activation function element-wise."""
    
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], derivative: Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.func = func
        self.derivative = derivative

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_data = X
        return self.func(X)

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        local_gradient = self.derivative(self.input_data)
        # Chain rule: element-wise multiplication
        return upstream_gradient * local_gradient

    def extra_repr(self) -> str:
        func_name = self.func.__name__ if hasattr(self.func, '__name__') else 'CustomFunction'
        return f"func={func_name}"

# ==============================================================================
# =============================== DROPOUT LAYER ==============================
# ==============================================================================
class Dropout(NNLayer):
    """
    Randomly zeroes some of the elements of the input tensor with probability p.
    Applies Inverted Dropout scaling to maintain signal strength.
    """
    def __init__(self, p: float = 0.5, rng: np.random.Generator = np.random.default_rng()):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        
        self.p = p
        self.rng = rng
        self.mask = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            # During evaluation, just pass the data through untouched
            return X
            
        # 1. Create a random boolean mask (1 with probability 1-p, 0 with probability p)
        # We use a binomial distribution to generate the 0s and 1s
        self.mask = self.rng.binomial(1, 1 - self.p, size=X.shape)
        
        # 2. Apply Inverted Dropout scaling
        scaling_factor = 1.0 / (1.0 - self.p)
        self.mask = self.mask * scaling_factor
        
        # 3. Apply the mask to the input
        return X * self.mask

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return upstream_gradient
            
        # The gradient only flows back through the neurons that were active!
        return upstream_gradient * self.mask

    def extra_repr(self) -> str:
        return f"p={self.p}"


# ==============================================================================
# =============================== 2D CONVOLUTION LAYER =========================
# ==============================================================================
class Conv2D(NNLayer):
    """
    A 2D Convolutional Layer for processing images.
    Expects input shape: (Batch_Size, In_Channels, Height, Width)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rng: np.random.Generator = np.random.default_rng()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # The weights (filters) shape: (Out_Channels, In_Channels, Kernel_Height, Kernel_Width)
        # Using He initialization suited for ReLU
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in)
        
        # Bias shape: (Out_Channels, 1) - one bias per filter
        self.bias = np.zeros((out_channels, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Slides the filters across the input image.
        X shape: (N, C_in, H_in, W_in)
        """
        self.input_data = X
        N, C_in, H_in, W_in = X.shape
        
        # Calculate output dimensions
        H_out = H_in - self.kernel_size + 1
        W_out = W_in - self.kernel_size + 1
        
        # Prepare the output array: (N, C_out, H_out, W_out)
        self.output = np.zeros((N, self.out_channels, H_out, W_out))
        
        # Slide the window across the spatial dimensions
        for i in range(H_out):
            for j in range(W_out):
                # Extract the local patch from all images in the batch
                # shape: (N, C_in, Kernel, Kernel)
                X_patch = X[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                
                # Multiply patch by all filters and sum. 
                # tensordot does this cleanly across the channel and kernel axes.
                self.output[:, :, i, j] = np.tensordot(
                    X_patch, self.weights, 
                    axes=([1, 2, 3], [1, 2, 3])
                )
        
        # Add the bias to each output channel
        # We reshape bias to (1, C_out, 1, 1) to broadcast correctly across the batch and spatial dims
        return self.output + self.bias.reshape(1, self.out_channels, 1, 1)

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Calculates gradients for weights, biases, and the previous layer.
        upstream_gradient shape: (N, C_out, H_out, W_out)
        """
        X = self.input_data
        N, C_in, H_in, W_in = X.shape
        _, C_out, H_out, W_out = upstream_gradient.shape
        
        # Initialize gradients
        self.grad_weights = np.zeros_like(self.weights)
        dX = np.zeros_like(X)
        
        # Gradient for bias: sum across batch, height, and width
        self.grad_bias = np.sum(upstream_gradient, axis=(0, 2, 3)).reshape(self.out_channels, 1)
        
        # Slide the window again to route the gradients back to the right weights and pixels
        for i in range(H_out):
            for j in range(W_out):
                X_patch = X[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                grad_patch = upstream_gradient[:, :, i, j] # shape: (N, C_out)
                
                # Update weight gradients (how much did this patch contribute to the error?)
                for n in range(N):
                    for c_out in range(C_out):
                        self.grad_weights[c_out] += X_patch[n] * grad_patch[n, c_out]
                        
                        # Update input gradients (pass the error back down the network)
                        dX[n, :, i:i+self.kernel_size, j:j+self.kernel_size] += self.weights[c_out] * grad_patch[n, c_out]
                        
        return dX

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}"

    def get_parameters(self) -> dict:
        return {'weights': self.weights.copy(), 'bias': self.bias.copy()}

    def set_parameters(self, params: dict):
        self.weights = params['weights']
        self.bias = params['bias']


# ==============================================================================
# =============================== FLATTEN LAYER ==============================
# ==============================================================================
class Flatten(NNLayer):
    """Flattens a multi-dimensional array into a 2D matrix (Batch, Features)."""
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_shape = X.shape
        # Flatten all dimensions except the batch size (index 0)
        return X.reshape(X.shape[0], -1)

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        # Reshape the flat gradient back into the original 3D/4D shape
        return upstream_gradient.reshape(self.input_shape)


# ==============================================================================
# =============================== 2D POOL LAYER ==============================
# ==============================================================================
class MaxPool2D(NNLayer):
    """Downsamples spatial dimensions by taking the maximum value in a window."""
    def __init__(self, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_data = X
        N, C, H, W = X.shape
        self.H_out = H // self.pool_size
        self.W_out = W // self.pool_size
        out = np.zeros((N, C, self.H_out, self.W_out))

        for i in range(self.H_out):
            for j in range(self.W_out):
                h_start, w_start = i * self.pool_size, j * self.pool_size
                patch = X[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                out[:, :, i, j] = np.max(patch, axis=(2, 3))
        return out

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        X = self.input_data
        dX = np.zeros_like(X)

        for i in range(self.H_out):
            for j in range(self.W_out):
                h_start, w_start = i * self.pool_size, j * self.pool_size
                patch = X[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                
                # Find which pixel was the maximum (creates a boolean mask of 1s and 0s)
                max_val = np.max(patch, axis=(2, 3), keepdims=True)
                mask = (patch == max_val)
                
                # Route the gradient ONLY to the pixel that "won" the max pooling
                grad = upstream_gradient[:, :, i, j][:, :, None, None]
                dX[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] += mask * grad
                
        return dX

    def extra_repr(self) -> str:
        return f"pool_size={self.pool_size}"