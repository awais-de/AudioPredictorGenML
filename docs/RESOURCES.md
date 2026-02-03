# Research & Learning Resources

Complete reading list organized by topic and day. Use this as your reference guide.

---

## **Day 1: DSP Crash Course**

### Must-Read:
1. **Oppenheim & Schafer** — *Discrete-Time Signal Processing* (3rd Edition)
   - Chapters 1-4: Sampling, frequency representation, DFT/FFT
   - [Amazon](https://www.amazon.com/Discrete-Time-Signal-Processing-3rd/dp/0131988751)
   - [ResearchGate PDF](https://www.researchgate.net/publication/236346975_Discrete-Time_Signal_Processing_3rd_Edition)

2. **MIT OpenCourseWare: 6.003 Signals and Systems** (Prof. Alan V. Oppenheim)
   - Lectures 1-6: Fundamental concepts, Fourier analysis
   - [Videos & Slides](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/)
   - [Lecture Notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/)

### Supplementary:
3. **3Blue1Brown** — "But what is the Fourier Transform?"
   - Intuitive visual explanation of FFT
   - [YouTube](https://www.youtube.com/watch?v=spUNpVPf6II)

4. **ProakisManolakis** — *Digital Signal Processing: Principles, Algorithms, and Applications* (4th)
   - [Amazon](https://www.amazon.com/Digital-Signal-Processing-Principles-Algorithms/dp/0131873741)

---

## **Day 2: Linear Prediction (LPC)**

### Must-Read:
1. **Makhoul, J.** (1975) — "Linear Prediction: A Tutorial Review"
   - Classic foundational paper on LPC
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/1454873)
   - [PDF Mirror](https://pdfs.semanticscholar.org/d3a3/84db6c6e6d9e8c2c5e1a3f7b9d4c8e2f1a.pdf)

2. **Rabiner & Schafer** — *Theory and Applications of Digital Speech Processing*
   - Chapter 8: Linear Predictive Coding (LPC)
   - [Amazon](https://www.amazon.com/Theory-Applications-Digital-Speech-Processing/dp/0135135796)
   - [Book PDF](http://ftp.cvut.cz/ftp/pub/courses/10E35ETP/)

### Supplementary:
3. **Schroeder & Atal** (1970) — "Code-Excited Linear Prediction (CELP)"
   - Extension of LPC for speech coding
   - [IEEE](https://ieeexplore.ieee.org/document/27949)

4. **Kay, S.** — *Modern Spectral Estimation*
   - AR models and spectral analysis
   - [Amazon](https://www.amazon.com/Modern-Spectral-Estimation-Theory-Applications/dp/0134641221)

---

## **Day 3: Lossless Audio Coding**

### Must-Read:
1. **Jayant & Noll** — *Digital Coding of Waveforms: Principles and Applications in Speech and Video*
   - Chapters 2-3: Predictive coding, quantization
   - [Amazon](https://www.amazon.com/Digital-Coding-Waveforms-Principles-Applications/dp/0133092208)

2. **FLAC Format Specification** — Xiph.Org Foundation
   - Practical lossless audio codec design
   - [Official Docs](https://xiph.org/flac/format.html)
   - [GitHub](https://github.com/xiph/flac)

### Supplementary:
3. **MPEG-4 ALS** — Audio Lossless Coding Standard
   - [ISO/IEC 14496-3](https://en.wikipedia.org/wiki/MPEG-4_Part_3#MPEG-4_ALS)
   - [Technical Paper](https://ieeexplore.ieee.org/document/6042177)

4. **Cover & Thomas** — *Elements of Information Theory* (Ch. 5)
   - Entropy and source coding theorems
   - [Amazon](https://www.amazon.com/Elements-Information-Theory-2nd/dp/0471241954)

---

## **Day 4: PyTorch Essentials**

### Must-Read:
1. **PyTorch Official Tutorials** — 60-Minute Blitz
   - Tensors, autograd, neural networks
   - [Official Link](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

2. **PyTorch Data Loading Tutorial**
   - Datasets, DataLoaders, custom data handling
   - [Official Docs](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

3. **Paszke et al.** — *Deep Learning with PyTorch: Build, Train, and Deploy* (Book)
   - Comprehensive PyTorch guide
   - [Amazon](https://www.amazon.com/Deep-Learning-PyTorch-Eli-Stevens/dp/1491912588)
   - [GitHub Code](https://github.com/deep-learning-with-pytorch/dlwpt-code)

### Supplementary:
4. **PyTorch Lightning** — Higher-level training framework
   - [Docs](https://pytorch-lightning.readthedocs.io/)

---

## **Day 5: Causal CNN & Dilated Convolutions**

### Must-Read:
1. **Bai et al.** (2018) — "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
   - Temporal Convolutional Networks (TCN)
   - [arXiv](https://arxiv.org/abs/1803.01271)
   - [GitHub](https://github.com/locuslab/TCN)

2. **van den Oord et al.** (2016) — "WaveNet: A Generative Model for Raw Audio"
   - Dilated causal convolutions, autoregressive models
   - [Paper](https://arxiv.org/abs/1609.03499)
   - [GitHub (DeepMind)](https://github.com/deepmind/wavenet)

### Supplementary:
3. **Dilated Convolutions Explained** — Dilip Krishnan (DeepMind Blog)
   - Visual intuition for receptive fields
   - [Blog Post](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)

4. **Krizhevsky et al.** (2012) — "ImageNet Classification with Deep CNNs"
   - Foundational CNN architecture concepts
   - [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

---

## **Day 6: Baseline Experiments & Evaluation**

### Must-Read:
1. **PyTorch Tutorial** — "Training a Classifier"
   - Adapt for regression tasks
   - [Official](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

2. **WaveNet Implementation Blogs** (Community)
   - High-level architecture walkthroughs
   - [Medium Article](https://medium.com/analytics-vidhya/understanding-wavenet-9fecf7f8e0f6)
   - [Fast AI Blog](https://www.fast.ai/2018/04/09/Lesson7.html) (related)

3. **Audio Quality Assessment** — ITU-R BS.1387 (PEAQ)
   - [Standard Doc](https://www.itu.int/rec/R-REC-BS.1387/en)

### Supplementary:
4. **Bengio et al.** (2013) — "Practical Recommendations for Gradient-Based Training"
   - Hyperparameter tuning, optimization strategies
   - [Paper](https://arxiv.org/abs/1206.5533)

---

## **Day 7: Autoregressive Generation**

### Must-Read:
1. **WaveNet Paper** (van den Oord et al., 2016) — Pages 5-6
   - Autoregressive sampling, temperature scaling
   - [Paper](https://arxiv.org/abs/1609.03499)

2. **FFTNet: A Real-Time GPU-Based Speech Synthesis** (Jin et al., 2018)
   - Fast autoregressive generation
   - [Paper](https://arxiv.org/abs/1811.06292)
   - [GitHub](https://github.com/gfickel/FFTNet)

### Supplementary:
3. **Grangier & Auli** (2017) — "Exploring the Limits of Transfer Learning"
   - Text-to-speech with autoregressive models
   - [Paper](https://arxiv.org/abs/1709.02230)

4. **Sampling Strategies** — Yoav Goldberg NLP Perspective
   - Temperature, top-k, top-p sampling
   - [Blog](https://nbviewer.jupyter.org/github/yoavg/ml_arxiv_ablations/blob/master/notebooks/sampling_strategies.ipynb)

---

## **Background Knowledge (Prerequisite)**

### Information Theory:
- **Cover & Thomas** — *Elements of Information Theory* (2nd Edition)
  - [Amazon](https://www.amazon.com/Elements-Information-Theory-2nd/dp/0471241954)
- **MacKay** — *Information Theory, Inference, and Learning Algorithms*
  - [Free PDF](http://www.inference.org.uk/itprnn/book.html)

### Machine Learning Fundamentals:
- **Goodfellow, Bengio, Courville** — *Deep Learning*
  - [Online Book](http://www.deeplearningbook.org/)
  - [Amazon](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618)
- **Bishop** — *Pattern Recognition and Machine Learning*
  - [Amazon](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)

### Sequence Modeling:
- **Hochreiter & Schmidhuber** (1997) — "Long Short-Term Memory"
  - LSTM cells and training RNNs
  - [Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **Vaswani et al.** (2017) — "Attention Is All You Need"
  - Transformer architecture (for context)
  - [Paper](https://arxiv.org/abs/1706.03762)

---

## **Advanced Topics (Learn As Needed)**

### Generative Models:
- **Goodfellow et al.** (2014) — "Generative Adversarial Nets"
  - [Paper](https://arxiv.org/abs/1406.2661)
- **Kingma & Welling** (2014) — "Variational Autoencoders"
  - [Paper](https://arxiv.org/abs/1312.6114)

### Quantization & Compression:
- **Bengio et al.** (2018) — "Quantized Neural Networks"
  - [Paper](https://arxiv.org/abs/1609.07061)

### Audio Datasets:
- **Ardila et al.** (2020) — "Common Voice: A Massively-Multilingual Speech Corpus"
  - [Paper](https://arxiv.org/abs/1912.06670)
- **MUSDB18** — Music Separation Database
  - [Dataset](https://sigsep.github.io/datasets/musdb.html)
- **VCTK Corpus** — Speech Synthesis Database
  - [Dataset](https://www.kaggle.com/nltkdata/vctk-corpus)

---

## **Quick Reference by Resource Type**

### Papers (arXiv & IEEE):
- WaveNet: https://arxiv.org/abs/1609.03499
- FFTNet: https://arxiv.org/abs/1811.06292
- TCN: https://arxiv.org/abs/1803.01271
- LPC Tutorial: https://ieeexplore.ieee.org/document/1454873

### Books:
- Oppenheim & Schafer (DSP)
- Goodfellow et al. (Deep Learning)
- Jayant & Noll (Audio Coding)

### Websites & Documentation:
- PyTorch: https://pytorch.org/
- FLAC Codec: https://xiph.org/flac/
- MIT OCW: https://ocw.mit.edu/

### Datasets:
- Common Voice, MUSDB18, VCTK, ESC-50, LibriSpeech

---

**Notes:**
- Access academic papers via ResearchGate, arXiv, or your institution
- Many textbooks have free PDFs available through ResearchGate or authors' websites
- Start with free resources (arXiv, blogs, OCW) then purchase textbooks
- Watch YouTube lectures alongside reading for better understanding
