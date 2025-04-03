import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from ..models.base_model import BaseModel

class Trainer:
    """Model trainer class for neural networks.
    
    Provides functionality for training, validation, and testing of neural network models.
    """
    
    def __init__(self, model, config, device=None):
        """Initialize the trainer.
        
        Args:
            model: Model instance inheriting from BaseModel
            config: Configuration object or dictionary with training parameters
            device: Device to use for training (default: automatically determined)
        """
        assert isinstance(model, BaseModel), "Model must inherit from BaseModel"
        
        self.model = model
        self.config = config
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 0)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.optimizer_name = config.get('optimizer', 'adam').lower()
        self.scheduler_name = config.get('scheduler', None)
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        self.early_stopping_patience = config.get('early_stopping_patience', None)
        
        # Checkpoint configuration
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        self.save_best_only = config.get('save_best_only', True)
        self.save_freq = config.get('save_freq', 1)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Determine device to use
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self._initialize_optimizer()
        
        # Initialize learning rate scheduler
        self._initialize_scheduler()
        
        # Initialize criterion (loss function)
        self.criterion = self._get_criterion(config.get('criterion', 'cross_entropy'))
        
        # Initialize training stats
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
    def _initialize_optimizer(self):
        """Initialize the optimizer based on configuration."""
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler based on configuration."""
        if self.scheduler_name is None:
            self.scheduler = None
        elif self.scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 5),
                gamma=self.config.get('lr_gamma', 0.1)
            )
        elif self.scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs
            )
        elif self.scheduler_name == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('lr_factor', 0.1),
                patience=self.config.get('lr_patience', 2),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
    
    def _get_criterion(self, criterion_name):
        """Get the loss function based on name.
        
        Args:
            criterion_name: Name of the criterion to use
            
        Returns:
            Loss function
        """
        if criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'mse':
            return nn.MSELoss()
        elif criterion_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif criterion_name == 'nll':
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    
    def train(self, train_loader, val_loader=None):
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            
        Returns:
            Training history dictionary containing loss and accuracy metrics
        """
        print(f"Training on {self.device}")
        print(f"Number of training examples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Number of validation examples: {len(val_loader.dataset)}")
        
        # Initialize early stopping counter
        early_stopping_counter = 0
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            self.stats['train_loss'].append(train_loss)
            self.stats['train_acc'].append(train_acc)
            
            # Validate if validation loader is provided
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.stats['val_loss'].append(val_loss)
                self.stats['val_acc'].append(val_acc)
                
                # Check if current model is best
                if val_loss < self.stats['best_val_loss']:
                    self.stats['best_val_loss'] = val_loss
                    self.stats['best_val_acc'] = val_acc
                    self.stats['best_epoch'] = epoch
                    
                    # Save best model
                    if self.save_best_only:
                        self._save_checkpoint(epoch, is_best=True)
                    
                    # Reset early stopping counter
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
            
            # Update learning rate if using scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_loader:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save current learning rate
            self.stats['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save periodic checkpoint if not save_best_only
            if not self.save_best_only and epoch % self.save_freq == 0:
                self._save_checkpoint(epoch)
            
            # Print epoch summary
            time_taken = time.time() - start_time
            self._print_epoch_summary(epoch, train_loss, train_acc, 
                                     val_loss if val_loader else None, 
                                     val_acc if val_loader else None,
                                     time_taken)
            
            # Check early stopping
            if self.early_stopping_patience and early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        return self.stats
    
    def _train_epoch(self, train_loader, epoch):
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average training loss, training accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': 100. * correct / total})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, desc="Validation"):
        """Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            desc: Description for progress bar (default: "Validation")
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=desc)
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': total_loss / len(data_loader), 'acc': 100. * correct / total})
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filename = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "best_model.pt"
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.stats = checkpoint['stats']
        
        return checkpoint['epoch']
    
    def _print_epoch_summary(self, epoch, train_loss, train_acc, val_loss, val_acc, time_taken):
        """Print a summary of the epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss (or None)
            val_acc: Validation accuracy (or None)
            time_taken: Time taken for the epoch in seconds
        """
        summary = f"\nEpoch {epoch}/{self.num_epochs} - time: {time_taken:.2f}s"
        summary += f" - train_loss: {train_loss:.4f} - train_acc: {train_acc:.2f}%"
        
        if val_loss is not None and val_acc is not None:
            summary += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%"
        
        summary += f" - lr: {self.optimizer.param_groups[0]['lr']:.1e}"
        
        print(summary)
