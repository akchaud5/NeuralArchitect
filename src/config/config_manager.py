import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional

class ConfigManager:
    """Configuration manager for the project.
    
    Handles loading, saving, and accessing configuration parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None, default_config: Optional[Dict[str, Any]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
            default_config: Default configuration dictionary to use if no file is provided
        """
        self.config_path = config_path
        self.config = {}
        
        # Load configuration
        if config_path is not None and os.path.exists(config_path):
            self.load_config(config_path)
        elif default_config is not None:
            self.config = default_config
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            Dictionary containing the configuration
        """
        self.config_path = config_path
        file_ext = Path(config_path).suffix.lower()
        
        try:
            with open(config_path, 'r') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    self.config = yaml.safe_load(f)
                elif file_ext == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
            
            print(f"Loaded configuration from {config_path}")
            return self.config
        
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            return {}
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to a file.
        
        Args:
            config_path: Path where to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            print("No config path specified for saving.")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            file_ext = Path(config_path).suffix.lower()
            with open(config_path, 'w') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    yaml.dump(self.config, f, default_flow_style=False)
                elif file_ext == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
            
            print(f"Saved configuration to {config_path}")
            return True
        
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the configuration.
        
        Supports nested keys with dot notation (e.g., 'model.learning_rate').
        
        Args:
            key: Key to retrieve (supports dot notation for nested dicts)
            default: Default value to return if key is not found
            
        Returns:
            Value associated with the key, or default if not found
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested keys
        current = self.config
        for part in key.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the configuration.
        
        Supports nested keys with dot notation (e.g., 'model.learning_rate').
        
        Args:
            key: Key to set (supports dot notation for nested dicts)
            value: Value to set
        """
        if '.' not in key:
            self.config[key] = value
            return
        
        # Handle nested keys
        parts = key.split('.')
        current = self.config
        
        # Navigate to the nested dictionary
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            
            current = current[part]
        
        # Set the value in the nested dictionary
        current[parts[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with values from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values to update
        """
        self._update_nested(self.config, config_dict)
    
    def _update_nested(self, current: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries.
        
        Args:
            current: Current dictionary to update
            update_dict: Dictionary with values to update
        """
        for key, value in update_dict.items():
            if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                self._update_nested(current[key], value)
            else:
                current[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.
        
        Returns:
            Dictionary containing all configuration values
        """
        return self.config


def get_default_config() -> Dict[str, Any]:
    """Create and return default configuration.
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        'general': {
            'seed': 42,
            'debug': False,
            'device': 'cuda',  # Options: 'cuda', 'cpu', 'auto'
            'checkpoint_dir': './checkpoints',
            'output_dir': './outputs',
            'log_dir': './logs',
        },
        'data': {
            'dataset': 'mnist',  # Options: 'mnist', 'cifar10', 'custom'
            'data_dir': './data',
            'batch_size': 32,
            'num_workers': 4,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
        },
        'model': {
            'type': 'cnn',  # Options: 'cnn', 'transformer', 'mlp', 'custom'
            'input_channels': 3,
            'num_classes': 10,
            'pretrained': False,
            'freeze_backbone': False,
            'dropout_rate': 0.5,
        },
        'cnn': {
            'conv_channels': [32, 64, 128],
            'fc_units': [512, 128],
        },
        'transformer': {
            'vocab_size': 10000,
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'max_seq_length': 512,
        },
        'training': {
            'num_epochs': 50,
            'optimizer': 'adam',  # Options: 'adam', 'sgd', 'adamw'
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'momentum': 0.9,  # For SGD
            'criterion': 'cross_entropy',  # Options: 'cross_entropy', 'mse', 'bce'
            'scheduler': 'cosine',  # Options: 'step', 'cosine', 'reduce_on_plateau', None
            'lr_step_size': 10,  # For StepLR scheduler
            'lr_gamma': 0.1,  # For StepLR scheduler
            'early_stopping_patience': 5,
            'save_best_only': True,
            'clip_grad_norm': 1.0,
        },
        'augmentation': {
            'use_augmentation': True,
            'horizontal_flip': True,
            'vertical_flip': False,
            'rotate': True,
            'scale': 0.2,
            'crop': True,
            'normalize': True,
            'cutout': False,
            'mixup': False,
            'mixup_alpha': 0.2,
        },
        'logging': {
            'print_frequency': 10,
            'save_frequency': 5,
            'log_to_file': True,
            'log_to_tensorboard': True,
            'tensorboard_update_frequency': 10,
        }
    }
