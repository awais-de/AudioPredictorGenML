# SSH/Remote Server Setup Guide

This guide helps you run AudioPredictorGenML on a GPU instance (e.g., AWS EC2, Lambda Labs, GCP, etc.)

## Quick Start on GPU Instance

### 1. **SSH into GPU Instance**

```bash
ssh -i your_key.pem ubuntu@your_gpu_instance.com
```

### 2. **Clone or Transfer Project**

**Option A: Clone from GitHub** (if you pushed)
```bash
git clone https://github.com/yourusername/AudioPredictorGenML.git
cd AudioPredictorGenML
```

**Option B: Transfer via SCP** (from local machine)
```bash
# On local machine
scp -r -i your_key.pem ./AudioPredictorGenML ubuntu@your_gpu_instance.com:/home/ubuntu/
```

### 3. **Setup Python Environment on GPU Instance**

```bash
# On GPU instance
cd AudioPredictorGenML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. **Start Training**

```bash
# Test configuration
python src/utils/config.py

# Run first experiment
python src/train/train.py --config configs/day4_toy.yaml \
  --device cuda \
  --exp_id exp_001_gpu_test

# Monitor in real-time
tail -f experiments/exp_001_gpu_test/logs/train.log
```

## Environment Variables for Remote Execution

Control execution via environment variables (useful for scripts/automation):

```bash
# Use GPU or CPU
export AUDIO_DEVICE=cuda        # or 'cpu'

# Number of data loading workers
export AUDIO_NUM_WORKERS=4

# Python logging level
export LOGLEVEL=INFO
```

Example:
```bash
AUDIO_DEVICE=cuda AUDIO_NUM_WORKERS=8 \
python src/train/train.py --config configs/day5_causal_cnn.yaml
```

## Running Experiments in Background

### Using `nohup` (Simple)

```bash
# Start training in background, logs to file
nohup python src/train/train.py --config configs/exp_config.yaml \
  --exp_id exp_002_long_run \
  > experiments/exp_002_long_run/logs/console.log 2>&1 &

# Detach and close SSH session safely
# Press Ctrl+Z, then 'bg', then close terminal
```

### Using `tmux` (Better for Long Runs)

```bash
# Create new tmux session
tmux new-session -d -s training

# Start training in tmux
tmux send-keys -t training \
  'cd AudioPredictorGenML && source venv/bin/activate && \
   python src/train/train.py --config configs/exp_config.yaml' C-m

# Monitor progress
tmux attach -t training

# Detach without stopping: Press Ctrl+B, then D
```

### Using `screen` (Alternative)

```bash
screen -S training
python src/train/train.py --config configs/exp_config.yaml

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

## Monitoring Remote Training

### Watch Logs in Real-Time

```bash
# From SSH terminal
tail -f experiments/exp_001/logs/train.log

# From local machine (over SSH)
ssh -i key.pem user@gpu.com \
  'tail -f AudioPredictorGenML/experiments/exp_001/logs/train.log'
```

### Check Metrics Periodically

```bash
# View current results
cat experiments/exp_001/results/metrics.json | python -m json.tool

# Download results locally
scp -r -i key.pem user@gpu.com:AudioPredictorGenML/experiments/exp_001 ./
```

### Monitor GPU Usage

```bash
# On GPU instance
watch -n 1 nvidia-smi

# Or one-time check
nvidia-smi
```

## SSH Tunneling for Jupyter Notebooks

Run interactive notebooks on GPU instance:

```bash
# On GPU instance, start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 \
  --no-browser sandbox/notebooks/

# On local machine, setup tunnel
ssh -L 8888:localhost:8888 -i key.pem user@gpu.com

# On local, open browser
# http://localhost:8888
# (copy token from GPU terminal)
```

## Data Management

### Upload Data to GPU Instance

```bash
# From local machine
scp -r data/*.wav user@gpu.com:AudioPredictorGenML/data/

# Or use rsync (faster for large files)
rsync -avz --progress data/ user@gpu.com:AudioPredictorGenML/data/
```

### Download Results Back to Local

```bash
# From local machine
rsync -avz user@gpu.com:AudioPredictorGenML/experiments ./

# Or specific experiment
scp -r user@gpu.com:AudioPredictorGenML/experiments/exp_001 ./results/
```

## Common Issues & Fixes

### 1. **CUDA Out of Memory**
```bash
# Use CPU for testing
export AUDIO_DEVICE=cpu
python src/train/train.py --config configs/day4_toy.yaml

# Or reduce batch size in config
```

### 2. **SSH Connection Timeout**

Add to local `~/.ssh/config`:
```
Host gpu_instance
    HostName your_gpu_instance.com
    User ubuntu
    IdentityFile ~/.ssh/your_key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 1440
```

Then: `ssh gpu_instance`

### 3. **Missing Dependencies on GPU Instance**

```bash
# Reinstall with specific PyTorch CUDA version
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Experiment Management Over SSH

### Create Experiment Template Script

```bash
# experiments/run_exp.sh
#!/bin/bash

EXP_ID=$1
CONFIG=$2

echo "Starting experiment: $EXP_ID"
python src/train/train.py \
    --config "$CONFIG" \
    --exp_id "$EXP_ID" \
    --device cuda

echo "Experiment complete: $EXP_ID"
```

Usage:
```bash
bash experiments/run_exp.sh exp_003_test configs/day5_causal_cnn.yaml
```

## Backup Strategy

Since you're on GPU instances (ephemeral), backup results:

```bash
# Download all experiments periodically
rsync -avz user@gpu.com:AudioPredictorGenML/experiments ./backups/$(date +%Y%m%d)/

# Or to cloud storage
aws s3 sync experiments/ s3://your-bucket/AudioPredictorGenML/experiments/
```

## Performance Tips

1. **Use Pinned Memory** (in config)
   ```yaml
   pin_memory: true
   ```

2. **Mixed Precision Training** (for speed on V100/A100)
   ```yaml
   mixed_precision: true
   ```

3. **Multiple GPU Workers**
   ```bash
   export AUDIO_NUM_WORKERS=8
   ```

4. **Optimize Data Loading**
   - Keep audio files in SSD (not EBS if possible)
   - Pre-process data on GPU instance, not locally

## Example Full Workflow

```bash
# 1. SSH to GPU instance
ssh -i key.pem ubuntu@gpu.com

# 2. Setup project
git clone https://github.com/you/AudioPredictorGenML.git
cd AudioPredictorGenML
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Upload data
scp data/*.wav user@gpu.com:AudioPredictorGenML/data/

# 4. Start training in tmux
tmux new-session -d -s exp1
tmux send-keys -t exp1 'source venv/bin/activate && \
  python src/train/train.py --config configs/day5_causal_cnn.yaml \
  --exp_id exp_001_gpu_run' C-m

# 5. Monitor from local machine
ssh -i key.pem ubuntu@gpu.com \
  'tail -f AudioPredictorGenML/experiments/exp_001_gpu_run/logs/train.log'

# 6. Download results when complete
rsync -avz ubuntu@gpu.com:AudioPredictorGenML/experiments/exp_001_gpu_run ./results/
```

---

**All code is SSH-compatible. No GUI dependencies. Ready for GPU!**
