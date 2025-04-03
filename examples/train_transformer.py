import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer_model import TransformerModel
from src.utils.trainer import Trainer
from src.data.datasets import TextDataset
from src.config.config_manager import ConfigManager, get_default_config

# Mock tokenizer for demo purposes
class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        
    def __call__(self, text, max_length=128, padding='max_length', truncation=True, return_tensors=None):
        # This is a very simplified tokenization for demonstration purposes
        # In a real scenario, you would use a proper tokenizer (e.g., from Hugging Face)
        tokens = [hash(word) % (self.vocab_size - 4) + 4 for word in text.split()]
        
        # Add special tokens: 0=PAD, 1=BOS, 2=EOS, 3=UNK
        tokens = [1] + tokens + [2]  # Add BOS and EOS tokens
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [2]  # Keep EOS token
        
        # Pad if necessary
        if padding == 'max_length':
            tokens = tokens + [0] * (max_length - len(tokens))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 0 else 0 for token in tokens]
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([tokens]),
                'attention_mask': torch.tensor([attention_mask])
            }
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }


def generate_mock_dataset(num_samples, seq_length, vocab_size):
    """Generate a mock text dataset for demonstration purposes."""
    sequences = []
    for _ in range(num_samples):
        # Generate random "text" (just space-separated numbers as words)
        num_words = np.random.randint(5, seq_length // 2)
        words = [str(np.random.randint(1, vocab_size)) for _ in range(num_words)]
        text = ' '.join(words)
        sequences.append(text)
    
    return sequences


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Transformer model on a text dataset')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    return parser.parse_args()


def collate_batch(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # For transformer training, we use the input shifted right as target
    target = input_ids[:, 1:].contiguous()
    source = input_ids[:, :-1].contiguous()
    
    return source, target


def main():
    """Main function for training a Transformer model on a text dataset."""
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
    
    # Define tokenizer and dataset parameters
    vocab_size = config['transformer']['vocab_size']
    max_seq_length = config['transformer']['max_seq_length']
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    # Generate mock dataset
    print("Generating mock dataset...")
    all_texts = generate_mock_dataset(
        num_samples=10000,
        seq_length=max_seq_length,
        vocab_size=vocab_size
    )
    
    # Create dataset
    dataset = TextDataset(all_texts, tokenizer, max_length=max_seq_length)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model_config = {
        'vocab_size': vocab_size,
        'd_model': config['transformer']['d_model'],
        'nhead': config['transformer']['nhead'],
        'num_encoder_layers': config['transformer']['num_encoder_layers'],
        'num_decoder_layers': config['transformer']['num_decoder_layers'],
        'dim_feedforward': config['transformer']['dim_feedforward'],
        'dropout': config['model']['dropout_rate'],
        'max_seq_length': max_seq_length
    }
    
    model = TransformerModel(model_config)
    print(f"Created Transformer model with {model.get_parameter_count():,} trainable parameters")
    
    # Create trainer
    trainer_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'num_epochs': config['training']['num_epochs'],
        'batch_size': config['data']['batch_size'],
        'optimizer': config['training']['optimizer'],
        'scheduler': config['training']['scheduler'],
        'criterion': 'cross_entropy',  # Override for language modeling
        'clip_grad_norm': config['training']['clip_grad_norm'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'checkpoint_dir': config['general']['checkpoint_dir'],
        'save_best_only': config['training']['save_best_only']
    }
    
    # Custom forward method for the trainer
    original_forward = model.forward
    
    def train_forward(x):
        # Unpack source and target from input
        src, tgt = x
        # Call the original forward method
        output = original_forward(src, tgt)
        # Reshape output for cross-entropy loss
        batch_size, seq_len, vocab_size = output.size()
        return output.reshape(batch_size * seq_len, vocab_size)
    
    # Monkey patch the forward method for training
    model.forward = train_forward
    
    trainer = Trainer(model, trainer_config, device)
    
    # Custom batch preprocessor for the trainer
    def preprocess_batch(batch, target):
        # Target is already part of the batch for the transformer
        # Just need to reshape it to match the reshaped predictions
        target = target.reshape(-1)  # Flatten target to match reshaped output
        return batch, target
    
    # Train the model
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    stats = trainer.train(train_loader, val_loader)
    
    # Print best results
    print(f"Best validation loss: {stats['best_val_loss']:.4f} (epoch {stats['best_epoch']})")
    print(f"Best validation accuracy: {stats['best_val_acc']:.2f}%")
    
    # Restore the original forward method
    model.forward = original_forward
    
    print("Training completed!")


if __name__ == '__main__':
    main()
