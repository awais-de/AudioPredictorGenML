"""LPC (Linear Predictive Coding) baseline predictor for Day 2."""

import numpy as np
from scipy import signal
import argparse
import json


class LPCPredictor:
    """Linear Predictive Coding predictor using Yule-Walker equations."""
    
    def __init__(self, order=16):
        """Initialize LPC predictor with given filter order."""
        self.order = order
        self.coeffs = None
    
    def fit(self, audio_signal):
        """Estimate LPC coefficients from audio signal."""
        # Compute autocorrelation using Yule-Walker method
        r = np.correlate(audio_signal, audio_signal, mode='full')
        r = r[len(r)//2:]  # Keep positive lags
        r = r[:self.order + 1]
        
        # Levinson-Durbin recursion
        a = np.zeros(self.order + 1)
        a[0] = 1.0
        e = r[0]
        
        for i in range(1, self.order + 1):
            alpha = -np.dot(a[1:i], r[i:0:-1]) / e
            a[1:i+1] = a[1:i] + alpha * a[i-1:0:-1]
            e *= (1 - alpha**2)
        
        self.coeffs = a
        return self
    
    def predict(self, audio_signal):
        """Predict next samples using fitted LPC model."""
        predictions = np.zeros_like(audio_signal)
        
        for n in range(self.order, len(audio_signal)):
            predictions[n] = -np.dot(self.coeffs[1:], audio_signal[n-self.order:n][::-1])
        
        return predictions
    
    def get_residual(self, audio_signal):
        """Compute prediction error (residual)."""
        predictions = self.predict(audio_signal)
        residual = audio_signal - predictions
        return residual
    
    def get_stats(self, audio_signal):
        """Compute statistics for evaluation."""
        residual = self.get_residual(audio_signal)
        mse = np.mean(residual[self.order:]**2)
        entropy = self._estimate_entropy(residual[self.order:])
        
        return {
            'order': self.order,
            'mse_residual': float(mse),
            'entropy_residual': float(entropy),
            'mean_residual': float(np.mean(residual)),
            'std_residual': float(np.std(residual))
        }
    
    @staticmethod
    def _estimate_entropy(signal, bins=256):
        """Estimate differential entropy using histogram binning."""
        hist, _ = np.histogram(signal, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0


def load_audio(filepath, sr=16000):
    """Load audio file using scipy."""
    from scipy.io import wavfile
    fs, audio = wavfile.read(filepath)
    
    # Resample if necessary
    if fs != sr:
        num_samples = int(len(audio) * sr / fs)
        audio = signal.resample(audio, num_samples)
    
    # Normalize to [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    return audio, sr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPC Predictor for Day 2')
    parser.add_argument('--audio_file', type=str, default=None)
    parser.add_argument('--order', type=int, default=16)
    parser.add_argument('--sr', type=int, default=16000)
    
    args = parser.parse_args()
    
    # If no audio file, generate synthetic test signal
    if args.audio_file:
        audio, sr = load_audio(args.audio_file, args.sr)
        print(f"Loaded audio: {args.audio_file} ({len(audio)} samples, {sr}Hz)")
    else:
        # Generate synthetic test signal (sum of sine waves)
        t = np.arange(16000) / 16000.0
        audio = (np.sin(2 * np.pi * 440 * t) + 
                 0.5 * np.sin(2 * np.pi * 880 * t) +
                 0.25 * np.random.randn(len(t)))
        sr = 16000
        print(f"Generated synthetic audio ({len(audio)} samples, {sr}Hz)")
    
    # Fit and evaluate LPC
    predictor = LPCPredictor(order=args.order)
    predictor.fit(audio)
    
    stats = predictor.get_stats(audio)
    print(f"\nLPC Predictor Results (order={args.order}):")
    for key, val in stats.items():
        print(f"  {key}: {val:.6f}")
    
    # Save stats
    output_file = 'results/lpc_stats.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {output_file}")
