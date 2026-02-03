"""
setup.py - Cross-platform environment setup for AudioPredictorGenML
Compatible with Windows, Linux, and macOS
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

    @classmethod
    def disable(cls):
        """Disable colors on Windows if not supported"""
        if platform.system() == 'Windows':
            cls.BLUE = cls.GREEN = cls.YELLOW = cls.RED = cls.NC = ''


def print_colored(text, color=Colors.NC):
    """Print colored text"""
    print(f"{color}{text}{Colors.NC}")


def print_header(text):
    """Print section header"""
    print_colored(f"\n{'='*50}", Colors.BLUE)
    print_colored(text, Colors.BLUE)
    print_colored('='*50, Colors.BLUE)


def run_command(cmd, description, silent=False, check=True):
    """Run shell command with error handling"""
    try:
        if silent:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
        return None
    except subprocess.CalledProcessError as e:
        print_colored(f"✗ {description} failed", Colors.RED)
        print_colored(f"Error: {e}", Colors.RED)
        if check:
            sys.exit(1)
        return None


def check_python_version():
    """Verify Python version >= 3.9"""
    print_colored("[1/8] Checking Python version...", Colors.YELLOW)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_colored(f"✗ Python 3.9+ required. Found: {version_str}", Colors.RED)
        sys.exit(1)
    
    print_colored(f"✓ Python {version_str} detected", Colors.GREEN)


def create_virtual_environment():
    """Create Python virtual environment"""
    print_colored("\n[2/8] Creating virtual environment...", Colors.YELLOW)
    
    venv_path = Path("venv")
    if venv_path.exists():
        print_colored("⚠ venv/ already exists. Skipping creation.", Colors.YELLOW)
        return
    
    run_command(f"{sys.executable} -m venv venv", "Virtual environment creation")
    print_colored("✓ Virtual environment created", Colors.GREEN)


def get_python_executable():
    """Get path to virtual environment Python executable"""
    system = platform.system()
    if system == "Windows":
        return Path("venv") / "Scripts" / "python.exe"
    else:
        return Path("venv") / "bin" / "python"


def get_activate_command():
    """Get activation command for virtual environment"""
    system = platform.system()
    if system == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def upgrade_pip(python_exe):
    """Upgrade pip, setuptools, and wheel"""
    print_colored("\n[3/8] Upgrading pip...", Colors.YELLOW)
    
    run_command(
        f"{python_exe} -m pip install --upgrade pip setuptools wheel",
        "pip upgrade",
        silent=True
    )
    print_colored("✓ pip upgraded", Colors.GREEN)


def install_dependencies(python_exe):
    """Install requirements from requirements.txt"""
    print_colored("\n[4/8] Installing dependencies from requirements.txt...", Colors.YELLOW)
    print_colored("This may take 5-10 minutes depending on your connection...", Colors.BLUE)
    
    run_command(
        f"{python_exe} -m pip install -r requirements.txt",
        "Dependency installation"
    )
    print_colored("✓ Dependencies installed", Colors.GREEN)


def verify_pytorch(python_exe):
    """Verify PyTorch installation and GPU availability"""
    print_colored("\n[5/8] Verifying PyTorch installation...", Colors.YELLOW)
    
    # Check PyTorch version
    version = run_command(
        f'{python_exe} -c "import torch; print(torch.__version__)"',
        "PyTorch verification",
        silent=True
    )
    print(f"PyTorch {version} installed")
    
    # Check GPU availability
    gpu_available = run_command(
        f'{python_exe} -c "import torch; exit(0 if torch.cuda.is_available() else 1)"',
        "GPU check",
        silent=True,
        check=False
    )
    
    if gpu_available == 0:
        gpu_name = run_command(
            f'{python_exe} -c "import torch; print(torch.cuda.get_device_name(0))"',
            "GPU name",
            silent=True
        )
        print_colored(f"✓ GPU detected: {gpu_name}", Colors.GREEN)
    else:
        print_colored("⚠ No GPU detected. CPU-only mode.", Colors.YELLOW)


def create_directory_structure():
    """Create required project directories"""
    print_colored("\n[6/8] Creating directory structure...", Colors.YELLOW)
    
    directories = [
        "data",
        "experiments",
        "sandbox/notebooks",
        "sandbox/scripts",
        "sandbox/scratch_code",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_colored("✓ Directories created", Colors.GREEN)


def verify_project_files():
    """Verify all required project files exist"""
    print_colored("\n[7/8] Verifying project files...", Colors.YELLOW)
    
    required_files = [
        "src/baseline/lpc_predictor.py",
        "src/models/causal_cnn.py",
        "src/train/train.py",
        "src/eval/metrics.py",
        "src/utils/config.py",
        "configs/day4_toy.yaml",
        "configs/day5_causal_cnn.yaml",
        "requirements.txt",
        ".gitignore",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_colored("✗ Missing files:", Colors.RED)
        for file_path in missing_files:
            print_colored(f"  - {file_path}", Colors.RED)
        sys.exit(1)
    
    print_colored("✓ All required files present", Colors.GREEN)


def print_success_message():
    """Print final success message with next steps"""
    print_colored("\n" + "="*50, Colors.GREEN)
    print_colored("✓ Setup completed successfully!", Colors.GREEN)
    print_colored("="*50, Colors.GREEN)
    
    activate_cmd = get_activate_command()
    
    print_colored("\nNext steps:", Colors.BLUE)
    print_colored(f"  1. Activate environment:  {activate_cmd}", Colors.YELLOW)
    print_colored("  2. Test DSP utilities:    python src/utils/dsp_basics.py --signal sine", Colors.YELLOW)
    print_colored("  3. Run LPC baseline:      python src/baseline/lpc_predictor.py", Colors.YELLOW)
    print_colored("  4. Start learning path:   Open docs/20260203_LEARNING_PATH.md", Colors.YELLOW)
    
    print_colored("\nFor GPU training on remote server:", Colors.BLUE)
    print_colored("  See docs/20260203_SSH_REMOTE_GUIDE.md\n", Colors.YELLOW)


def main():
    """Main setup routine"""
    # Disable colors on Windows if needed
    if platform.system() == 'Windows':
        Colors.disable()
    
    print_header("AudioPredictorGenML - Environment Setup")
    
    try:
        # Setup steps
        check_python_version()
        create_virtual_environment()
        
        python_exe = get_python_executable()
        
        upgrade_pip(python_exe)
        install_dependencies(python_exe)
        verify_pytorch(python_exe)
        create_directory_structure()
        verify_project_files()
        
        print_success_message()
        
    except KeyboardInterrupt:
        print_colored("\n\n✗ Setup interrupted by user", Colors.RED)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n✗ Unexpected error: {e}", Colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    main()
