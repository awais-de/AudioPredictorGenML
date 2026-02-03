"""Generate audio autoregressively from trained model."""

import torch
import numpy as np
import argparse
from pathlib import Path


def generate_audio(model, length=16000, batch_size=1, temperature=1.0, 
                  seed=42, device='cpu'):
    """Generate audio autoregressively."""
    torch.manual_seed(seed)
    model.eval()
    
    # Initialize with small random values
    receptive_field = model.receptive_field
    generated = torch.randn(batch_size, 1, receptive_field, 
                           device=device) * 0.01
    
    with torch.no_grad():
        for i in range(length):
            # Predict next sample
            pred = model(generated[:, :, -receptive_field:])
            
            # Sample with temperature
            if temperature > 0:
                noise = torch.randn_like(pred[:, :, -1:]) * temperature
            else:
                noise = 0
            
            next_sample = pred[:, :, -1:] + noise
            
            # Append to sequence
            generated = torch.cat([generated, next_sample], dim=2)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i+1}/{length} samples")
    
    return generated[:, :, receptive_field:].cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio from trained model')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--length', type=int, default=16000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='generated.wav')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    print(f"This is a template. Implement actual generation logic here.")
