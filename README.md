# AudioPredictorGenML

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Type-Academic%20Research-brightgreen.svg)]()

**Deep Generative Model for Audio Prediction and Synthesis using Machine Learning**

> *A Masters-level research project investigating neural network-based predictors for lossless audio coding and autoregressive synthesis*

## Abstract

This research project explores the application of deep neural networks (DNNs) and convolutional neural networks (CNNs) to audio signal prediction for lossless audio coding. We implement and evaluate causal CNN architectures inspired by WaveNet and FFTNet, comparing their performance against classical Linear Predictive Coding (LPC) baselines. The project investigates autoregressive generation through feedback loops, analyzing prediction residuals, compression gains, and synthesis quality. Our framework provides SSH-compatible training infrastructure for GPU acceleration, comprehensive experiment tracking, and reproducible evaluation metrics.

**Keywords**: Audio prediction · Deep learning · Lossless coding · WaveNet · Causal convolutions · Linear predictive coding · Autoregressive generation

---

## 1. Project Overview

- **Research Goal**: Design and evaluate ML-based audio predictors (DNN, CNN) that outperform state-of-the-art in lossless audio coding
- **Secondary Goal**: Investigate autoregressive generation using feedback loops and analyze synthesis quality
- **Academic Level**: Masters thesis (20 ECTS credits, ~600 hours)
- **Supervision**: Prof. Gerald Schuller (TU Ilmenau), Dr.-Ing. Sascha Disch, Dipl.-Math. Andreas Niedermeier (Fraunhofer IIS)
- **Tech Stack**: Python 3.9+, PyTorch 2.1+, librosa 0.10+, torchaudio, scipy, numpy
- **Infrastructure**: Fully SSH-compatible for remote GPU deployment (AWS, Lambda Labs, GCP)

---

## 2. Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/awais-de/AudioPredictorGenML.git
cd AudioPredictorGenML

# Run setup script (creates environment, installs dependencies)
bash setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Verification

```bash
# Test DSP utilities
python src/utils/dsp_basics.py --signal sine

# Run baseline LPC predictor
python src/baseline/lpc_predictor.py

# Verify PyTorch GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Remote GPU Setup

For SSH-based GPU training on cloud instances (AWS, Lambda Labs, GCP):
1. SSH to your instance: `ssh -i key.pem user@instance.com`
2. Clone repository and run setup: `bash setup.sh`
3. Set environment variable: `export AUDIO_DEVICE=cuda`
4. Start training with GPU acceleration

---

---

## 3. Methodology

### 3.1 Architecture Overview

```
Audio Signal → Predictor → Residual → Entropy Coding → Compression
                  ↑                         ↓
                  └─── Feedback Loop ───────┘ (Autoregressive)
```

### 3.2 Baseline: Linear Predictive Coding (LPC)

Classical signal processing baseline using Levinson-Durbin recursion:
- Autocorrelation-based coefficient estimation
- Parametric modeling of spectral envelope
- Serves as minimum performance threshold

**Implementation**: [src/baseline/lpc_predictor.py](src/baseline/lpc_predictor.py)

### 3.3 Causal CNN Predictor

WaveNet-inspired architecture with:
- **Dilated causal convolutions**: Exponentially increasing receptive field without future information
- **Residual connections**: Improved gradient flow for deep networks
- **Gated activations**: `tanh × sigmoid` for expressive capacity
- **Autoregressive generation**: Sequential sample-by-sample synthesis

**Implementation**: [src/models/causal_cnn.py](src/models/causal_cnn.py)

**Key Parameters**:
- Input channels: 1 (raw audio)
- Hidden channels: 64-128
- Layers: 8-12 (receptive field: ~1024-4096 samples)
- Kernel size: 3
- Dilation: [1, 2, 4, 8, 16, 32, 64, 128, ...]

### 3.4 Training Procedure

- **Loss**: Mean Squared Error (MSE) on prediction residuals
- **Optimizer**: Adam (lr=1e-3, weight decay=1e-5)
- **Batch size**: 32-64 (depending on GPU memory)
- **Sequence length**: 8192 samples (~185ms at 44.1kHz)
- **Early stopping**: Patience=10 epochs on validation loss
- **Teacher forcing**: 100% during training

**Implementation**: [src/train/train.py](src/train/train.py)

---

## 4. Experimental Setup

### 4.1 Datasets

- **Initial**: LibriSpeech (speech, 16kHz) for proof-of-concept
- **Target**: VCTK Corpus, NSynth (musical instruments)
- **Preprocessing**: Normalization to [-1, 1], zero-padding for batching

### 4.2 Evaluation Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **MSE** | Mean Squared Error (sample-level accuracy) | [metrics.py](src/eval/metrics.py) |
| **MAE** | Mean Absolute Error (robust to outliers) | [metrics.py](src/eval/metrics.py) |
| **SNR** | Signal-to-Noise Ratio (dB) | [metrics.py](src/eval/metrics.py) |
| **Entropy** | Residual entropy (bits/sample) | [metrics.py](src/eval/metrics.py) |
| **Compression Gain** | Original vs predicted entropy ratio | [metrics.py](src/eval/metrics.py) |

### 4.3 Hardware

- **Development**: Laptop with CPU (M1/Intel)
- **Training**: GPU instances (NVIDIA T4/V100/A100)
- **SSH-Compatible**: Lambda Labs, AWS EC2, GCP Compute Engine

---

## 5. Results

> *This section will be populated during experimental phases*

### 5.1 LPC Baseline

| Order | MSE | SNR (dB) | Entropy (bits/sample) |
|-------|-----|----------|------------------------|
| 8     | TBD | TBD      | TBD                    |
| 16    | TBD | TBD      | TBD                    |

### 5.2 Causal CNN Predictor

| Layers | Receptive Field | MSE | SNR (dB) | Training Time |
|--------|----------------|-----|----------|---------------|
| 8      | 1024           | TBD | TBD      | TBD           |
| 12     | 4096           | TBD | TBD      | TBD           |

### 5.3 Autoregressive Generation Quality

- **Perceptual Quality**: TBD (subjective listening tests)
- **Synthesis Speed**: TBD (samples/second)
- **Divergence Analysis**: TBD (long-term stability)

---

---

## 6. Repository Structure

```
AudioPredictorGenML/
├── src/                      # Production code (Git-tracked)
│   ├── baseline/             # LPC baseline predictor
│   │   └── lpc_predictor.py
│   ├── models/               # Neural network models
│   │   ├── causal_cnn.py
│   │   └── generate.py
│   ├── train/                # Training utilities & loops
│   │   └── train.py
│   ├── eval/                 # Evaluation metrics
│   │   └── metrics.py
│   └── utils/                # DSP utilities, config management
│       ├── dsp_basics.py
│       └── config.py
│
├── experiments/              # Experiment tracking
│   ├── exp_001_lpc_baseline/
│   │   ├── config.yaml       # Experiment configuration
│   │   ├── metadata.json     # Results & metadata
│   │   ├── checkpoints/      # Model weights (*.pth)
│   │   ├── logs/             # Training/validation logs
│   │   ├── results/          # Metrics & outputs
│   │   └── artifacts/        # Generated audio files
│   └── README_EXPERIMENTS.md # Experiment tracking guide
│
├── sandbox/                  # Exploratory learning space
│   ├── notebooks/            # Jupyter notebooks
│   ├── scripts/              # Test scripts
│   ├── scratch_code/         # Experimental code
│   └── README_SANDBOX.md     # Sandbox usage guide
│
├── docs/                     # Public documentation
│   ├── RESOURCES.md          # Research papers & references
│   └── SSH_REMOTE_GUIDE.md   # GPU setup guide
│
├── configs/                  # YAML experiment configs
│   ├── day4_toy.yaml
│   └── day5_causal_cnn.yaml
│
├── data/                     # Dataset location
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules
├── LICENSE                   # MIT License
├── CITATION.cff              # Citation metadata
└── README.md                 # This file
```

---

## 7. Running Experiments

### Local Training

```bash
# Activate environment
source venv/bin/activate

# Train toy model (quick test)
python src/train/train.py --config configs/day4_toy.yaml \
  --exp_id exp_001_local_test

# Results saved to experiments/exp_001_local_test/
```

### Remote GPU Training

```bash
# SSH to GPU instance
ssh -i ~/.ssh/key.pem user@gpu.instance.com

# Clone repository & setup
git clone https://github.com/awais-de/AudioPredictorGenML.git
cd AudioPredictorGenML
bash setup.sh

# Start training with GPU
AUDIO_DEVICE=cuda python src/train/train.py \
  --config configs/day5_causal_cnn.yaml \
  --exp_id exp_002_gpu_run \
  --gpu_id 0

# Monitor from local machine
ssh user@gpu.instance.com \
  "tail -f ~/AudioPredictorGenML/experiments/exp_002_gpu_run/logs/train.log"
```

### Experiment Tracking

Each experiment creates isolated folder:
```
experiments/exp_001_local_test/
├── config.yaml          # Copy of experiment config
├── metadata.json        # Results, metrics, timestamps
├── checkpoints/
│   ├── epoch_010.pth
│   ├── epoch_020.pth
│   └── best_model.pth
├── logs/
│   ├── train.log
│   └── tensorboard/
├── results/
│   ├── metrics.json
│   └── predictions.wav
└── artifacts/
    └── generated_samples.wav
```

---

## 8. Key References

### Foundational Papers

1. **WaveNet**: van den Oord et al., "WaveNet: A Generative Model for Raw Audio", arXiv:1609.03499 (2016)
2. **FFTNet**: Jin et al., "FFTNet: A Real-Time Speaker-Dependent Neural Vocoder", ICASSP (2018)
3. **Linear Prediction**: Makhoul, "Linear Prediction: A Tutorial Review", Proceedings of the IEEE (1975)
4. **Lossless Coding**: Sayood, "Introduction to Data Compression", 5th Edition

### Textbooks

- Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3rd Edition
- Goodfellow et al., *Deep Learning*, MIT Press (2016)
- Rabiner & Schafer, *Theory and Applications of Digital Speech Processing*

### Full Bibliography

See [docs/RESOURCES.md](docs/RESOURCES.md) for complete curated references

---

## 9. Project Timeline

| Phase | Weeks | Focus | Deliverables |
|-------|-------|-------|--------------|
| **1. Foundations** | 1-2 | DSP, LPC, PyTorch basics | Baseline implementations |
| **2. Implementation** | 2-4 | Model design & coding | Causal CNN implementation |
| **3. Training** | 5-6 | Experiments & optimization | Trained models, metrics |
| **4. Generation** | 7-8 | Autoregressive synthesis | Generated audio samples |
| **5. Refinement** | 9-12 | Scaling, hyperparameter tuning | Production models |
| **6. Documentation** | 13-15 | Paper writing & submission | Thesis submission |

**Total**: 600 hours (~15 weeks full-time)

---

## 10. Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{AudioPredictorGenML2026,
  author = {Muhammad Awais},
  title = {Deep Generative Model for Audio Prediction and Synthesis using Machine Learning},
  school = {Technische Universität Ilmenau},
  year = {2026},
  type = {Masters Thesis},
  note = {Supervised by Prof. Gerald Schuller, Dr.-Ing. Sascha Disch, Dipl.-Math. Andreas Niedermeier}
}
```

See [CITATION.cff](CITATION.cff) for structured citation metadata.

---

## 11. Acknowledgments

This research project is conducted under the supervision of:

- **Prof. Gerald Schuller** — Technische Universität Ilmenau, Institute for Media Technology
- **Dr.-Ing. Sascha Disch** — Fraunhofer Institute for Integrated Circuits IIS, Audio and Media Technologies
- **Dipl.-Math. Andreas Niedermeier** — Fraunhofer Institute for Integrated Circuits IIS

Special thanks to the open-source community for PyTorch, librosa, and related audio processing libraries.

---

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Muhammad Awais

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full MIT License text...]
```

---

## 13. Contributing

This is an academic research project for Masters thesis completion. External contributions are not accepted during the active research period. After thesis submission, the repository may be opened for community contributions.

---

## 14. Contact

- **Student**: Muhammad Awais — m.awais@tu-ilmenau.de
- **Supervisor**: Prof. Gerald Schuller — gerald.schuller@tu-ilmenau.de

---

**Ready to start? Clone the repository and run `bash setup.sh` to begin! 🚀**
