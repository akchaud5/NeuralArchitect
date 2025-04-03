# Neural Networks Project

A comprehensive PyTorch-based neural networks framework designed for implementation, experimentation, and deployment of various neural network architectures. The project provides a modular structure to support both educational purposes and real-world applications.

## Overview

This framework enables researchers and developers to:

- Implement and experiment with different neural network architectures
- Train models on various data types (images, text, time series)
- Configure training parameters through a flexible configuration system
- Track and visualize training metrics and model performance
- Deploy trained models for inference

## Features

- **Modular Architecture**: Easily extendable framework for implementing various neural network architectures
- **Multiple Model Support**: Ready-to-use implementations of CNNs and Transformers
- **Configuration Management**: Flexible configuration system using YAML/JSON files
- **Training Pipeline**: Robust training utilities with support for early stopping, checkpointing, and various optimizers/schedulers
- **Metrics Tracking**: Comprehensive metrics computation and tracking for model evaluation
- **Dataset Abstractions**: Implementations for different data types (images, text, time series)
- **Visualization Tools**: Utilities for visualizing training progress, model predictions, and performance metrics

## Supported Models

- **CNNs**: Configurable convolutional neural networks for image classification tasks
- **Transformers**: Encoder-decoder transformer architecture for text generation and sequence modeling

## Supported Datasets

The framework has been tested with the following datasets:

- **Image Classification**: MNIST, CIFAR-10
- **Language Modeling**: WikiText-2, Mock text data
- **Time Series**: Supports custom time series data

## Project Structure

```
├── src/                  # Source code
│   ├── models/           # Neural network model implementations
│   │   ├── base_model.py # Abstract base class for all models
│   │   ├── cnn_model.py  # CNN implementation
│   │   └── transformer_model.py # Transformer implementation
│   ├── data/             # Data loading and processing utilities
│   │   └── datasets.py   # Dataset implementations
│   ├── utils/            # Utility functions
│   │   ├── trainer.py    # Training loop implementation
│   │   └── metrics.py    # Metrics computation utilities
│   └── config/           # Configuration management
│       └── config_manager.py # Configuration loading/saving
├── tests/                # Unit tests
│   └── test_models.py    # Tests for model implementations
├── examples/             # Example scripts
│   ├── train_cnn.py      # Example script for training a CNN
│   └── train_transformer.py # Example script for training a Transformer
├── notebooks/            # Jupyter notebooks for experiments
│   ├── mnist_cnn_tutorial.ipynb      # MNIST classification tutorial
│   ├── cifar10_cnn_training.ipynb    # CIFAR-10 classification
│   ├── transformer_text_generation.ipynb  # Basic text generation with transformer
│   └── wikitext_transformer.ipynb    # Language modeling on WikiText
└── docs/                 # Documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/nn-project.git
cd nn-project
```

2. Install the package and dependencies:

```bash
pip install -e .
```

## Quick Start

### Image Classification (CNN)

```python
from src.models.cnn_model import CNNModel
from src.utils.trainer import Trainer

# Define model configuration
model_config = {
    'input_channels': 3,  # RGB images
    'num_classes': 10,    # Number of output classes
    'conv_channels': [32, 64, 128],  # Channels in conv layers
    'fc_units': [512, 128],  # Units in fully-connected layers
    'dropout_rate': 0.5   # Dropout probability
}

# Create and train the model
model = CNNModel(model_config)
trainer = Trainer(model, trainer_config)
trainer.train(train_loader, val_loader)
```

### Text Generation (Transformer)

```python
from src.models.transformer_model import TransformerModel
from src.utils.trainer import Trainer

# Define model configuration
model_config = {
    'vocab_size': 10000,  # Vocabulary size
    'd_model': 512,       # Embedding dimension
    'nhead': 8,           # Number of attention heads
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'max_seq_length': 128
}

# Create and train the model
model = TransformerModel(model_config)
trainer = Trainer(model, trainer_config)
trainer.train(train_loader, val_loader)

# Generate text
generated = model.generate(input_sequence, max_length=100)
```

## Usage Examples

### Training a CNN on MNIST

```bash
python examples/train_cnn.py --epochs 10 --batch_size 64 --lr 0.001
```

### Training a Transformer model

```bash
python examples/train_transformer.py --epochs 5 --batch_size 32 --lr 0.0001
```

### Using a custom configuration file

```bash
python examples/train_cnn.py --config configs/my_config.yaml
```

## Jupyter Notebooks

The project includes several Jupyter notebooks to demonstrate different aspects of the framework:

- **MNIST CNN Tutorial**: Basic introduction to the framework using MNIST dataset
- **CIFAR-10 CNN Training**: More advanced CNN training with data augmentation and detailed analysis
- **Transformer Text Generation**: Text generation using a transformer model
- **WikiText Transformer**: Language modeling on the WikiText-2 dataset

## Extending the Framework

### Adding a New Model

Create a new model class that inherits from `BaseModel`:

```python
from src.models.base_model import BaseModel

class MyNewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model architecture
    
    def forward(self, x):
        # Implement the forward pass
        return output
```

### Adding a New Dataset

Create a new dataset class that inherits from `torch.utils.data.Dataset`:

```python
from torch.utils.data import Dataset

class MyNewDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Initialize your dataset
        
    def __len__(self):
        # Return the size of the dataset
        
    def __getitem__(self, idx):
        # Return a sample from the dataset
```

## Running Tests

```bash
python -m unittest discover tests
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- tqdm
- scikit-learn
- PyYAML
- Matplotlib
- (For certain notebooks): Hugging Face Transformers, Datasets

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.