import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class CNNModel(BaseModel):
    """Convolutional Neural Network implementation.
    
    A flexible CNN architecture that can be configured for various vision tasks.
    """
    
    def __init__(self, config):
        """Initialize CNN model.
        
        Args:
            config: Configuration object or dictionary with the following keys:
                - input_channels: Number of input channels
                - num_classes: Number of output classes
                - conv_channels: List of channel sizes for each conv layer
                - fc_units: List of hidden units for fully connected layers
                - dropout_rate: Dropout probability
        """
        super().__init__(config)
        
        # Configuration
        self.input_channels = config.get('input_channels', 3)
        self.num_classes = config.get('num_classes', 10)
        self.conv_channels = config.get('conv_channels', [32, 64, 128])
        self.fc_units = config.get('fc_units', [512, 128])
        self.dropout_rate = config.get('dropout_rate', 0.5)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        
        for out_channels in self.conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels
        
        # Calculate size of flattened features after conv layers
        # Assuming input size is at least 32x32 and we have 3 conv layers with maxpooling
        self.feature_size = (32 // (2**len(self.conv_channels)))**2 * self.conv_channels[-1]
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_units = self.feature_size
        
        for units in self.fc_units:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(prev_units, units),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ))
            prev_units = units
        
        # Output layer
        self.output_layer = nn.Linear(prev_units, self.num_classes)
        
    def forward(self, x):
        """Forward pass of the CNN model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
