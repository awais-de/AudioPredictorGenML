"""Causal CNN predictor for Day 5."""

import torch
import torch.nn as nn
import numpy as np


class CausalConv1d(nn.Module):
    """1D Causal Convolution: maintains causality by padding only on the left."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
    
    def forward(self, x):
        # Remove extra padding from right side to maintain causality
        return self.conv(x)[:, :, :-self.dilation*(self.kernel_size-1)]


class CausalCNNPredictor(nn.Module):
    """Causal CNN predictor with dilated convolutions."""
    
    def __init__(self, num_layers=4, num_filters=64, kernel_size=3, 
                 max_dilation=16, input_channels=1, output_channels=1):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        dilation = 1
        
        for i in range(num_layers):
            if i == 0:
                in_ch = input_channels
            else:
                in_ch = num_filters
            
            layers.append(CausalConv1d(in_ch, num_filters, kernel_size, 
                                      dilation=dilation))
            layers.append(nn.ReLU())
            
            # Increase dilation exponentially, then reset
            dilation = min(dilation * 2, max_dilation)
        
        # Output layer (1D prediction)
        layers.append(nn.Conv1d(num_filters, output_channels, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Calculate receptive field
        self.receptive_field = self._compute_receptive_field()
    
    def _compute_receptive_field(self):
        """Compute receptive field size."""
        rf = 1
        dilation = 1
        for _ in range(self.num_layers):
            rf += (3 - 1) * dilation
            dilation = min(dilation * 2, 16)
        return rf
    
    def forward(self, x):
        """Forward pass: x shape (batch, channels, length)."""
        return self.network(x)
    
    def generate(self, batch_size=1, length=1000, temperature=1.0, device='cpu'):
        """Generate audio autoregressively."""
        self.eval()
        
        # Initialize with random or zeros
        generated = torch.randn(batch_size, 1, self.receptive_field, 
                               device=device) * 0.1
        
        with torch.no_grad():
            for _ in range(length):
                # Predict next sample
                pred = self.forward(generated[:, :, -self.receptive_field:])
                
                # Sample from prediction (add noise)
                next_sample = pred[:, :, -1:] + torch.randn_like(pred[:, :, -1:]) * temperature
                
                # Append to sequence
                generated = torch.cat([generated, next_sample], dim=2)
        
        return generated.cpu().numpy()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Causal CNN Predictor')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    
    args = parser.parse_args()
    
    # Create model
    model = CausalCNNPredictor(num_layers=args.num_layers, 
                              num_filters=args.num_filters,
                              kernel_size=args.kernel_size)
    
    print(f"Model architecture:\n{model}")
    print(f"\nReceptive field: {model.receptive_field} samples")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 256)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
