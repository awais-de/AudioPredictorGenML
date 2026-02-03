#!/bin/bash
# setup.sh - Environment setup script for AudioPredictorGenML
# Compatible with Linux and macOS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AudioPredictorGenML - Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check Python version
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}\n"

# Create virtual environment
echo -e "${YELLOW}[2/8] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Warning: venv/ already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/8] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}\n"

# Upgrade pip
echo -e "${YELLOW}[4/8] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}\n"

# Install dependencies
echo -e "${YELLOW}[5/8] Installing dependencies from requirements.txt...${NC}"
echo -e "${BLUE}This may take 5-10 minutes depending on your connection...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Verify PyTorch installation
echo -e "${YELLOW}[6/8] Verifying PyTorch installation...${NC}"
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"

# Check GPU availability
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected. CPU-only mode.${NC}"
fi
echo ""

# Create directory structure
echo -e "${YELLOW}[7/8] Creating directory structure...${NC}"
mkdir -p data
mkdir -p experiments
mkdir -p sandbox/notebooks
mkdir -p sandbox/scripts
mkdir -p sandbox/scratch_code
mkdir -p tests
echo -e "${GREEN}✓ Directories created${NC}\n"

# Verify key files
echo -e "${YELLOW}[8/8] Verifying project files...${NC}"
REQUIRED_FILES=(
    "src/baseline/lpc_predictor.py"
    "src/models/causal_cnn.py"
    "src/train/train.py"
    "src/eval/metrics.py"
    "src/utils/config.py"
    "configs/day4_toy.yaml"
    "configs/day5_causal_cnn.yaml"
    "requirements.txt"
    ".gitignore"
    "README.md"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All required files present${NC}\n"
else
    echo -e "${RED}✗ Missing files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "${RED}  - $file${NC}"
    done
    echo ""
    exit 1
fi

# Final success message
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Activate environment:  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  2. Test DSP utilities:    ${YELLOW}python src/utils/dsp_basics.py --signal sine${NC}"
echo -e "  3. Run LPC baseline:      ${YELLOW}python src/baseline/lpc_predictor.py${NC}"
echo -e "  4. Start learning path:   ${YELLOW}Open docs/20260203_LEARNING_PATH.md${NC}\n"

echo -e "${BLUE}For GPU training on remote server:${NC}"
echo -e "  See ${YELLOW}docs/20260203_SSH_REMOTE_GUIDE.md${NC}\n"
