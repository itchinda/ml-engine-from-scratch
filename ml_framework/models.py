from .layers import NNLayer
import numpy as np

# ==============================================================================
# =============================== SEQUENTIAL LAYER ==============================
# ==============================================================================
class Sequential(NNLayer):
    """A container layer that passes data sequentially through a list of layers."""
    
    def __init__(self, layers: list[NNLayer]):
        super().__init__()
        self.layers = layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        activation = X
        for layer in self.layers:
            activation = layer(activation)
        return activation

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        gradient = upstream_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def get_parameters(self) -> dict:
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_parameters()
            # Only save if the layer actually has parameters
            if layer_params: 
                params[f'layer_{i}'] = layer_params
        return params

    def set_parameters(self, params: dict):
        for i, layer in enumerate(self.layers):
            if f'layer_{i}' in params:
                layer.set_parameters(params[f'layer_{i}'])