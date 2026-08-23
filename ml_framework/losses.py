import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

# ===================================================================================
# =================== ABSTRACT BASE CLASS FOR LOSS FUNCTIONS ========================
# ===================================================================================
class LossFunction(ABC):
    """Abstract base class for all loss functions."""
    
    def __init__(self):
        # Caches for the backward pass
        self.y_predicted: Optional[np.ndarray] = None
        self.y_expected: Optional[np.ndarray] = None

    @abstractmethod
    def __call__(self, y_predicted: np.ndarray, y_expected: np.ndarray) -> float:
        """Calculates and returns the scalar loss value."""
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        """Calculates and returns the gradient of the loss with respect to predictions."""
        pass

# ===================================================================================
# ================ MEAN SQUARED ERROR LOSS FOR REGRESSION PROBLEMS ==================
# ===================================================================================
class MSELoss(LossFunction):
    """Mean Squared Error (MSE) Loss for regression tasks."""
    
    def __call__(self, y_predicted: np.ndarray, y_expected: np.ndarray) -> float:
        """
        Calculates the scalar Mean Squared Error loss.
        Caches predictions and targets for backpropagation.
        """
        self.y_predicted = y_predicted
        self.y_expected = y_expected
        
        return float(np.mean((y_predicted - y_expected) ** 2))

    def backward(self) -> np.ndarray:
        """Returns the gradient of the loss with respect to y_predicted."""
        if self.y_predicted is None or self.y_expected is None:
            raise ValueError("You must compute the loss forward pass prior to performing backpropagation.")
        
        N = self.y_predicted.shape[0] 
        return (2 / N) * (self.y_predicted - self.y_expected)

# ===================================================================================
# ================ BINARY CROSS ENTOPY LOSS FOR MULTICLASS CLASSIFICATION ===========
# ===================================================================================
class BinaryCrossEntropyLoss(LossFunction):
    """
    Combines Sigmoid and Binary Cross-Entropy Loss for numerical stability.
    Ideal for binary (0 or 1) classification tasks.
    Expects raw logits as input (do not put a Sigmoid layer before this).
    """
    def __call__(self, logits: np.ndarray, y_expected: np.ndarray) -> float:
        # Apply numerically stable Sigmoid
        logits_clipped = np.clip(logits, -500, 500)
        self.y_predicted = 1.0 / (1.0 + np.exp(-logits_clipped))
        self.y_expected = y_expected
        
        # Calculate BCE Loss
        epsilon = 1e-15 # Prevent log(0)
        N = logits.shape[0]
        
        # BCE Formula
        loss = -np.sum(
            self.y_expected * np.log(self.y_predicted + epsilon) + 
            (1 - self.y_expected) * np.log(1 - self.y_predicted + epsilon)
        ) / N
        
        return float(loss)

    def backward(self) -> np.ndarray:
        """Returns the combined gradient of Sigmoid and BCE."""
        if self.y_predicted is None or self.y_expected is None:
            raise ValueError("You must call the forward pass before backward.")
            
        N = self.y_predicted.shape[0]
        
        # The magically simplified gradient!
        return (self.y_predicted - self.y_expected) / N
    
# ===================================================================================
# ================ CROSS ENTOPY LOSS FOR MULTICLASS CLASSIFICATION ==================
# ===================================================================================
class CrossEntropyLoss(LossFunction):
    """
    Fonction de perte combinant Softmax et l'Entropie Croisée (Cross-Entropy).
    Idéal pour les problèmes de classification multi-classes.
    """
    def __call__(self, logits: np.ndarray, y_expected: np.ndarray) -> float:
        """
        Calcule la perte. Attend des étiquettes en format One-Hot.
        """
        # 1. Softmax avec stabilité numérique (on soustrait le max pour éviter l'overflow de np.exp)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        self.y_predicted = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_expected = y_expected
        
        # 2. Calcul de l'Entropie Croisée
        # Ajout d'un minuscule epsilon (1e-15) pour éviter np.log(0)
        epsilon = 1e-15
        N = logits.shape[0]
        
        # Formule: -sum(Y_true * log(Y_pred)) / N
        loss = -np.sum(self.y_expected * np.log(self.y_predicted + epsilon)) / N
        
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Retourne le gradient de la perte (Softmax + Cross-Entropy combinés)
        par rapport aux logits bruts.
        """
        if self.y_predicted is None or self.y_expected is None:
            raise ValueError("Vous devez appeler la perte (forward) avant le backward.")
            
        N = self.y_predicted.shape[0]
        
        # The magically simplified gradient!
        return (self.y_predicted - self.y_expected) / N
