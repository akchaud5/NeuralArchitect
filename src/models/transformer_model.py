import torch
import torch.nn as nn
import math
from .base_model import BaseModel

class TransformerModel(BaseModel):
    """Transformer-based neural network model.
    
    A configurable transformer architecture suitable for sequence modeling tasks.
    """
    
    def __init__(self, config):
        """Initialize transformer model.
        
        Args:
            config: Configuration object or dictionary with the following keys:
                - vocab_size: Size of vocabulary
                - d_model: Dimension of embeddings and hidden states
                - nhead: Number of attention heads
                - num_encoder_layers: Number of encoder layers
                - num_decoder_layers: Number of decoder layers
                - dim_feedforward: Dimension of feedforward network
                - dropout: Dropout probability
                - max_seq_length: Maximum sequence length
                - pad_idx: Padding token index (optional)
        """
        super().__init__(config)
        
        # Configuration
        self.vocab_size = config.get('vocab_size', 10000)
        self.d_model = config.get('d_model', 512)
        self.nhead = config.get('nhead', 8)
        self.num_encoder_layers = config.get('num_encoder_layers', 6)
        self.num_decoder_layers = config.get('num_decoder_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.pad_idx = config.get('pad_idx', 0)
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Transformer encoder/decoder
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
    def _create_positional_encoding(self):
        """Create positional encodings for the transformer.
        
        Returns:
            Tensor of shape [1, max_seq_length, d_model] containing positional encodings
        """
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        
        pos_encoding = torch.zeros(1, self.max_seq_length, self.d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def _create_padding_mask(self, x):
        """Create padding mask for transformer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length]
            
        Returns:
            Boolean mask of shape [batch_size, seq_length] where True indicates padding positions
        """
        return x == self.pad_idx
    
    def forward(self, src, tgt):
        """Forward pass of transformer model.
        
        Args:
            src: Source sequence tensor of shape [batch_size, src_seq_length]
            tgt: Target sequence tensor of shape [batch_size, tgt_seq_length]
            
        Returns:
            Output tensor of shape [batch_size, tgt_seq_length, vocab_size]
        """
        # Create masks
        src_padding_mask = self._create_padding_mask(src)
        tgt_padding_mask = self._create_padding_mask(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Embed tokens and add positional encoding
        src_embedded = self.token_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.token_embedding(tgt) * math.sqrt(self.d_model)
        
        src_embedded = src_embedded + self.positional_encoding[:, :src_embedded.size(1), :].to(src.device)
        tgt_embedded = tgt_embedded + self.positional_encoding[:, :tgt_embedded.size(1), :].to(tgt.device)
        
        src_embedded = self.dropout_layer(src_embedded)
        tgt_embedded = self.dropout_layer(tgt_embedded)
        
        # Pass through transformer
        output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary size
        output = self.output_projection(output)
        
        return output
    
    def generate(self, src, max_length=100, temperature=1.0):
        """Generate sequence using trained transformer model.
        
        Args:
            src: Source sequence tensor of shape [batch_size, src_seq_length]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (1.0 = greedy)
            
        Returns:
            Generated sequence tensor of shape [batch_size, max_length]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Start with BOS token (assuming index 1)
        output_ids = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # Auto-regressive generation
        for i in range(max_length - 1):
            # Get model predictions
            with torch.no_grad():
                curr_output = self.forward(src, output_ids)
                next_token_logits = curr_output[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # Append to output sequence
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # Break if all sequences have reached EOS token (assuming index 2)
            if (next_token == 2).all():
                break
        
        return output_ids
