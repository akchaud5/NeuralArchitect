import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models.
    
    This class defines the interface that all model implementations should follow.
    It inherits from both PyTorch's nn.Module and Python's ABC to ensure proper
    implementation of required methods.
    """
    
    def __init__(self, config):
        """Initialize the base model.
        
        Args:
            config: Configuration object or dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output tensor
        """
        pass
    
    def save(self, path):
        """Save model weights to the specified path.
        
        Args:
            path: Path where model weights will be saved
        """
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load model weights from the specified path.
        
        Args:
            path: Path from which to load model weights
        """
        self.load_state_dict(torch.load(path))
        
    def get_parameter_count(self):
        """Count the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
