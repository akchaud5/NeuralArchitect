import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDataset(Dataset):
    """Dataset for loading image data.
    
    Supports loading images from a directory, with various transformations.
    """
    
    def __init__(self, root_dir, transform=None, split='train'):
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to apply to images
            split: Data split ('train', 'val', or 'test')
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.split = split
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)
        
        return img, label


class TextDataset(Dataset):
    """Dataset for loading text data.
    
    Supports loading text data for language modeling or sequence tasks.
    """
    
    def __init__(self, data, tokenizer, max_length=128):
        """Initialize the dataset.
        
        Args:
            data: List of text samples
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with tokenized inputs
        """
        text = self.data[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        for key in encoding.keys():
            encoding[key] = encoding[key].squeeze(0)
        
        return encoding


class TimeSeriesDataset(Dataset):
    """Dataset for time series data.
    
    Supports loading time series data with sliding window approach.
    """
    
    def __init__(self, data, seq_length, target_idx=None, transform=None):
        """Initialize the dataset.
        
        Args:
            data: Time series data as numpy array or torch tensor
            seq_length: Length of the sequence window
            target_idx: Index of the target variable(s)
            transform: Optional transform to apply to sequences
        """
        self.data = torch.tensor(data) if isinstance(data, np.ndarray) else data
        self.seq_length = seq_length
        self.target_idx = target_idx
        self.transform = transform
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (sequence, target)
        """
        # Get sequence window
        sequence = self.data[idx:idx+self.seq_length]
        
        # Apply transform if specified
        if self.transform:
            sequence = self.transform(sequence)
        
        # If target_idx is provided, use it to extract target variables
        if self.target_idx is not None:
            target = self.data[idx+self.seq_length, self.target_idx]
            return sequence, target
        else:
            # Otherwise use next timestep as target
            return sequence[:-1], sequence[-1]


def create_dataloaders(dataset, batch_size, split_ratio=(0.7, 0.15, 0.15), shuffle=True, num_workers=4, seed=42):
    """Create train, validation, and test dataloaders from a dataset.
    
    Args:
        dataset: PyTorch Dataset object
        batch_size: Batch size for dataloaders
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate split sizes
    train_size = int(split_ratio[0] * len(dataset))
    val_size = int(split_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
