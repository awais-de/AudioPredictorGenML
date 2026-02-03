"""Evaluation metrics and comparison for Day 6."""

import numpy as np
import json
from pathlib import Path


class AudioEvaluator:
    """Evaluate audio predictors on various metrics."""
    
    @staticmethod
    def mse_residual(audio, predictions):
        """Mean squared error of residuals."""
        residuals = audio - predictions
        return np.mean(residuals**2)
    
    @staticmethod
    def mae_residual(audio, predictions):
        """Mean absolute error of residuals."""
        residuals = audio - predictions
        return np.mean(np.abs(residuals))
    
    @staticmethod
    def entropy_residual(residuals, bins=256):
        """Estimate entropy of residual distribution."""
        hist, _ = np.histogram(residuals, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    @staticmethod
    def compression_gain(audio, residuals, bits_per_sample_original=16):
        """Estimate compression gain: (bits_original - bits_compressed) / bits_original."""
        entropy = AudioEvaluator.entropy_residual(residuals)
        bits_compressed = entropy  # Approximate bits per sample
        original_bits = bits_per_sample_original
        
        gain = (original_bits - bits_compressed) / original_bits
        return max(0, gain)  # Ensure non-negative
    
    @staticmethod
    def snr_db(audio, predictions):
        """Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(audio**2)
        noise_power = np.mean((audio - predictions)**2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def evaluate(audio, predictions, output_file=None):
        """Compute all metrics."""
        residuals = audio - predictions
        
        metrics = {
            'mse_residual': float(AudioEvaluator.mse_residual(audio, predictions)),
            'mae_residual': float(AudioEvaluator.mae_residual(audio, predictions)),
            'entropy_residual': float(AudioEvaluator.entropy_residual(residuals)),
            'compression_gain': float(AudioEvaluator.compression_gain(audio, residuals)),
            'snr_db': float(AudioEvaluator.snr_db(audio, predictions)),
        }
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return metrics


if __name__ == '__main__':
    print("Evaluation metrics module loaded successfully.")
