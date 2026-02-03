---
name: Prerequisites Checklist
about: Track prerequisite learning progress
title: "Prerequisites: Learning Checklist"
labels: ["prereq", "learning"]
assignees: ["awais-de"]
---

## Prerequisites Checklist

> Mark items as you complete them.

### 1) DSP Fundamentals
- [ ] Signal basics (time vs frequency domain) — https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/
- [ ] Fourier Transform (FFT) intuition — https://www.dspguide.com/ch8.htm
- [ ] Short-Time Fourier Transform (STFT) — https://librosa.org/doc/latest/generated/librosa.stft.html
- [ ] Filtering fundamentals — https://www.dspguide.com/ch14.htm
- [ ] Run: python [src/utils/dsp_basics.py](src/utils/dsp_basics.py) --signal sine

### 2) PyTorch Foundations
- [ ] Tensors and basic ops — https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- [ ] Autograd basics — https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- [ ] Simple MLP training loop — https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- [ ] Read: [src/models/causal_cnn.py](src/models/causal_cnn.py)

### 3) Linear Predictive Coding (LPC)
- [ ] Concept: prediction residuals — https://www.dspguide.com/ch12.htm
- [ ] Levinson–Durbin recursion (high-level understanding) — https://www.dspguide.com/ch12/5.htm
- [ ] Read: [src/baseline/lpc_predictor.py](src/baseline/lpc_predictor.py)

### 4) Core Reading (curated)
- [ ] WaveNet paper (van den Oord et al., 2016) — https://arxiv.org/abs/1609.03499
- [ ] FFTNet paper (Jin et al., 2018) — https://arxiv.org/abs/1802.08449
- [ ] Makhoul LPC tutorial (1975) — https://ieeexplore.ieee.org/document/1162949
- [ ] Skim: [docs/RESOURCES.md](docs/RESOURCES.md)

### Notes
- [ ] Add brief notes here (optional)

---

### Progress Log (optional)
- Date:
- What I learned:
- Open questions:
