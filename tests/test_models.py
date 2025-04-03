import sys
import os
import unittest
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.base_model import BaseModel
from src.models.cnn_model import CNNModel
from src.models.transformer_model import TransformerModel


class TestBaseModel(unittest.TestCase):
    """Test cases for the BaseModel class."""
    
    def test_abstract_methods(self):
        """Test that BaseModel requires implementation of abstract methods."""
        with self.assertRaises(TypeError):
            # Should raise TypeError because forward is an abstract method
            model = BaseModel({})
    
    def test_parameter_count(self):
        """Test the parameter counting functionality."""
        # Create a simple model inheriting from BaseModel for testing
        class SimpleModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel({})
        # Linear layer 10x5 should have 10*5 + 5 = 55 parameters
        self.assertEqual(model.get_parameter_count(), 55)


class TestCNNModel(unittest.TestCase):
    """Test cases for the CNNModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'input_channels': 3,
            'num_classes': 10,
            'conv_channels': [16, 32],
            'fc_units': [128],
            'dropout_rate': 0.5
        }
        self.model = CNNModel(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, CNNModel)
        self.assertIsInstance(self.model, BaseModel)
        
        # Check that model has the expected number of layers
        self.assertEqual(len(self.model.conv_layers), len(self.config['conv_channels']))
        self.assertEqual(len(self.model.fc_layers), len(self.config['fc_units']))
    
    def test_forward_pass(self):
        """Test forward pass with a random input tensor."""
        batch_size = 4
        height, width = 32, 32
        input_tensor = torch.randn(batch_size, self.config['input_channels'], height, width)
        
        output = self.model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.config['num_classes']))
        
        # Check that output has valid probabilities
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
            self.assertTrue((probs >= 0).all())
            self.assertTrue((probs <= 1).all())
            self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(batch_size)))
    
    def test_save_load(self):
        """Test saving and loading model weights."""
        tmp_path = '/tmp/cnn_model_test.pt'
        
        # Generate random weights
        for param in self.model.parameters():
            param.data = torch.randn_like(param.data)
        
        original_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Save model
        self.model.save(tmp_path)
        
        # Create a new model with random weights
        new_model = CNNModel(self.config)
        for param in new_model.parameters():
            param.data = torch.randn_like(param.data)
        
        # Load saved weights
        new_model.load(tmp_path)
        
        # Check that weights match
        for name, param in new_model.named_parameters():
            self.assertTrue(torch.allclose(param, original_weights[name]))
        
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class TestTransformerModel(unittest.TestCase):
    """Test cases for the TransformerModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'vocab_size': 1000,
            'd_model': 64,
            'nhead': 2,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'max_seq_length': 50
        }
        self.model = TransformerModel(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, TransformerModel)
        self.assertIsInstance(self.model, BaseModel)
        
        # Check positional encoding shape
        self.assertEqual(self.model.positional_encoding.shape, 
                         (1, self.config['max_seq_length'], self.config['d_model']))
    
    def test_forward_pass(self):
        """Test forward pass with random input tensors."""
        batch_size = 4
        src_seq_len = 10
        tgt_seq_len = 8
        
        # Create random token IDs
        src = torch.randint(0, self.config['vocab_size'], (batch_size, src_seq_len))
        tgt = torch.randint(0, self.config['vocab_size'], (batch_size, tgt_seq_len))
        
        output = self.model(src, tgt)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, tgt_seq_len, self.config['vocab_size']))
    
    def test_generate(self):
        """Test the generation functionality."""
        batch_size = 2
        src_seq_len = 10
        max_length = 15
        
        # Create random token IDs
        src = torch.randint(0, self.config['vocab_size'], (batch_size, src_seq_len))
        
        generated = self.model.generate(src, max_length=max_length)
        
        # Check that generation produces sequences of the expected shape
        self.assertEqual(generated.shape[0], batch_size)  # Batch dimension preserved
        self.assertLessEqual(generated.shape[1], max_length)  # Length constraint respected


if __name__ == '__main__':
    unittest.main()
