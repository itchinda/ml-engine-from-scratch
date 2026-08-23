import numpy as np
from abc import ABC, abstractmethod
from .layers import NNLayer

# ===============================================================================
# =================== ABSTRACT BASE CLASS FOR OPTIMIZERS ========================
# ===============================================================================
class Optim(ABC):
    """
        Abstract base class for optimizers
    """
    def __init__(self, learning_rate: float = 0.01 ):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layer: NNLayer) -> None:
        pass

# ===============================================================================
# =================== STOCHASTIC GRADIENT DESCENT OPTIMIZER =====================
# ===============================================================================
class SGD(Optim):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, layer: NNLayer):
        """Recursively traverses layers and updates learnable parameters."""
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                self.step(sub_layer)
        
        elif hasattr(layer, 'weights') and hasattr(layer, 'bias'):
            if layer.grad_weights is not None and layer.grad_bias is not None:
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias


# ===============================================================================
# =============================== ADAM'S OPTIMIZER ==============================
# ===============================================================================
class Adam(Optim):
    """Adaptive Moment Estimation optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 
        self.state = {}

    def step(self, model: NNLayer):
        self.t += 1
        self._update_layer(model)

    def _update_layer(self, layer: NNLayer):
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                self._update_layer(sub_layer)
                
        elif hasattr(layer, 'weights') and hasattr(layer, 'bias'):
            layer_id = id(layer)
            if layer_id not in self.state:
                self.state[layer_id] = {
                    'm_w': np.zeros_like(layer.weights),
                    'v_w': np.zeros_like(layer.weights),
                    'm_b': np.zeros_like(layer.bias),
                    'v_b': np.zeros_like(layer.bias)
                }
            
            s = self.state[layer_id]
            
            # Biased moments
            s['m_w'] = self.beta1 * s['m_w'] + (1 - self.beta1) * layer.grad_weights
            s['v_w'] = self.beta2 * s['v_w'] + (1 - self.beta2) * (layer.grad_weights ** 2)
            s['m_b'] = self.beta1 * s['m_b'] + (1 - self.beta1) * layer.grad_bias
            s['v_b'] = self.beta2 * s['v_b'] + (1 - self.beta2) * (layer.grad_bias ** 2)
            
            # Bias-corrected moments
            m_w_hat = s['m_w'] / (1 - self.beta1 ** self.t)
            v_w_hat = s['v_w'] / (1 - self.beta2 ** self.t)
            m_b_hat = s['m_b'] / (1 - self.beta1 ** self.t)
            v_b_hat = s['v_b'] / (1 - self.beta2 ** self.t)
            
            # Parameter updates
            layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
