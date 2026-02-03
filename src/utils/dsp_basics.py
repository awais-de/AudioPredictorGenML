"""DSP utilities for Day 1 — Crash course basics."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse


def generate_signal(signal_type='sine', freq=440, sr=16000, duration=2):
    """Generate test signal."""
    t = np.arange(int(sr * duration)) / sr
    if signal_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif signal_type == 'noise':
        return np.random.randn(len(t))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def compute_fft(signal, sr=16000):
    """Compute FFT and frequency bins."""
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sr)
    return np.abs(fft), freqs


def compute_stft(signal, sr=16000, n_fft=512, hop_length=256):
    """Compute STFT and spectrogram."""
    window = signal.hann(n_fft)  # Use Hann window
    f, t, Sxx = signal.spectrogram(signal, sr, window=window, 
                                     nperseg=n_fft, noverlap=n_fft-hop_length)
    return 10 * np.log10(Sxx + 1e-10), f, t  # Log scale


def plot_signal_analysis(signal, sr=16000, output_file=None):
    """Plot time-domain, FFT, and spectrogram."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time domain
    t = np.arange(len(signal)) / sr
    axes[0].plot(t, signal)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain')
    axes[0].grid()
    
    # FFT
    fft_mag, freqs = compute_fft(signal, sr)
    positive_idx = freqs[:len(freqs)//2] >= 0
    axes[1].plot(freqs[positive_idx], fft_mag[positive_idx])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('FFT')
    axes[1].set_xlim(0, sr/2)
    axes[1].grid()
    
    # Spectrogram
    spec, f, t_spec = compute_stft(signal, sr)
    im = axes[2].pcolormesh(t_spec, f, spec, shading='gouraud', cmap='viridis')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Spectrogram (dB)')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved: {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSP basics for Day 1')
    parser.add_argument('--signal', type=str, default='sine', 
                        choices=['sine', 'noise'])
    parser.add_argument('--freq', type=float, default=440)
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    # Generate and analyze signal
    sig = generate_signal(args.signal, args.freq, args.sr, args.duration)
    print(f"Generated {args.signal} signal: freq={args.freq}Hz, duration={args.duration}s")
    print(f"Signal shape: {sig.shape}, min={sig.min():.3f}, max={sig.max():.3f}")
    
    plot_signal_analysis(sig, args.sr, args.output)
