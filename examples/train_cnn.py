import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cnn_model import CNNModel
from src.utils.trainer import Trainer
from src.config.config_manager import ConfigManager, get_default_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a CNN model on MNIST dataset')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    return parser.parse_args()


def main():
    """Main function for training a CNN model on MNIST dataset."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize configuration
    config_manager = ConfigManager(args.config, get_default_config())
    config = config_manager.get_all()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config_manager.set('training.num_epochs', args.epochs)
    if args.batch_size is not None:
        config_manager.set('data.batch_size', args.batch_size)
    if args.lr is not None:
        config_manager.set('training.learning_rate', args.lr)
    if args.device is not None:
        config_manager.set('general.device', args.device)
    if args.output_dir is not None:
        config_manager.set('general.output_dir', args.output_dir)
    
    # Create output directory
    os.makedirs(config['general']['output_dir'], exist_ok=True)
    os.makedirs(config['general']['checkpoint_dir'], exist_ok=True)
    
    # Save updated configuration
    config_path = os.path.join(config['general']['output_dir'], 'config.yaml')
    config_manager.save_config(config_path)
    
    # Set device
    if config['general']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['general']['device'])
    
    # Set random seed for reproducibility
    torch.manual_seed(config['general']['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(config['general']['seed'])
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model_config = {
        'input_channels': 1,  # MNIST is grayscale
        'num_classes': 10,
        'conv_channels': config['cnn']['conv_channels'],
        'fc_units': config['cnn']['fc_units'],
        'dropout_rate': config['model']['dropout_rate']
    }
    
    model = CNNModel(model_config)
    print(f"Created CNN model with {model.get_parameter_count():,} trainable parameters")
    
    # Create trainer
    trainer_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'num_epochs': config['training']['num_epochs'],
        'batch_size': config['data']['batch_size'],
        'optimizer': config['training']['optimizer'],
        'scheduler': config['training']['scheduler'],
        'criterion': config['training']['criterion'],
        'clip_grad_norm': config['training']['clip_grad_norm'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'checkpoint_dir': config['general']['checkpoint_dir'],
        'save_best_only': config['training']['save_best_only']
    }
    
    trainer = Trainer(model, trainer_config, device)
    
    # Train the model
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    stats = trainer.train(train_loader, test_loader)
    
    # Print best results
    print(f"Best validation loss: {stats['best_val_loss']:.4f} (epoch {stats['best_epoch']})")
    print(f"Best validation accuracy: {stats['best_val_acc']:.2f}%")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
