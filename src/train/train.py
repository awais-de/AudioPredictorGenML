"""Training loop and utilities for Day 4-6."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path


class AudioPredictorTrainer:
    """Trainer class for audio predictor models."""
    
    def __init__(self, model, device='cpu', lr=1e-3, checkpoint_dir='results/checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc='Training')):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc='Validation'):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(self, train_loader, val_loader, epochs=10, patience=3):
        """Train with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Checkpoint loaded: {path}")


def create_sequences(signal, seq_length):
    """Create overlapping sequences for training."""
    x, y = [], []
    for i in range(len(signal) - seq_length):
        x.append(signal[i:i+seq_length])
        y.append(signal[i+seq_length:i+seq_length+1])
    
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def create_data_loaders(signal, seq_length=256, batch_size=32, 
                       train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test data loaders."""
    x, y = create_sequences(signal, seq_length)
    
    n_train = int(len(x) * train_ratio)
    n_val = int(len(x) * val_ratio)
    
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    x_test, y_test = x[n_train+n_val:], y[n_train+n_val:]
    
    # Convert to tensors
    train_data = TensorDataset(torch.from_numpy(x_train).unsqueeze(1),
                              torch.from_numpy(y_train).unsqueeze(1))
    val_data = TensorDataset(torch.from_numpy(x_val).unsqueeze(1),
                            torch.from_numpy(y_val).unsqueeze(1))
    test_data = TensorDataset(torch.from_numpy(x_test).unsqueeze(1),
                             torch.from_numpy(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Training utilities module loaded successfully.")
