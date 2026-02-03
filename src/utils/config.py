"""
Configuration utilities for SSH/remote server execution.
Supports remote paths, environment variables, and GPU setup.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging to file (important for remote execution)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager supporting:
    - Local and remote execution
    - Environment variables
    - GPU/CPU selection
    - Path management
    """
    
    def __init__(self, config_file: str = None):
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file or self.project_root / 'configs' / 'default.yaml'
        self.config = self._load_config()
        self._setup_paths()
        self._setup_environment()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or defaults."""
        try:
            import yaml
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            return self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'device': 'cuda' if self._has_gpu() else 'cpu',
            'num_workers': 4,
            'pin_memory': True,
            'mixed_precision': False,
        }
    
    @staticmethod
    def _has_gpu() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _setup_paths(self):
        """Setup all project paths (works locally and over SSH)."""
        # Use relative paths from project root
        self.paths = {
            'project_root': self.project_root,
            'src': self.project_root / 'src',
            'data': self.project_root / 'data',
            'experiments': self.project_root / 'experiments',
            'sandbox': self.project_root / 'sandbox',
            'configs': self.project_root / 'configs',
            'results': self.project_root / 'results',
        }
        
        # Create directories if they don't exist
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _setup_environment(self):
        """Setup environment variables for remote execution."""
        # Allow override from environment
        device = os.getenv('AUDIO_DEVICE', self.config.get('device', 'cuda'))
        num_workers = int(os.getenv('AUDIO_NUM_WORKERS', self.config.get('num_workers', 4)))
        
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = self.config.get('pin_memory', True)
        self.mixed_precision = self.config.get('mixed_precision', False)
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Num workers: {self.num_workers}")
    
    def get_path(self, key: str) -> Path:
        """Get path by key, works over SSH."""
        if key not in self.paths:
            raise KeyError(f"Path '{key}' not found. Available: {list(self.paths.keys())}")
        return self.paths[key]
    
    def create_experiment_dir(self, exp_id: str) -> Path:
        """Create experiment directory structure."""
        exp_dir = self.paths['experiments'] / exp_id
        subdirs = ['checkpoints', 'logs', 'results', 'artifacts']
        
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created experiment directory: {exp_dir}")
        return exp_dir
    
    def save_experiment_metadata(self, exp_id: str, metadata: Dict[str, Any]):
        """Save experiment metadata to JSON."""
        from datetime import datetime
        
        exp_dir = self.paths['experiments'] / exp_id
        metadata_file = exp_dir / 'metadata.json'
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {metadata_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'mixed_precision': self.mixed_precision,
            'paths': {k: str(v) for k, v in self.paths.items()},
        }


def setup_remote_training():
    """
    One-time setup for remote GPU training.
    Call this before starting training on GPU instance.
    """
    logger.info("Setting up remote training environment...")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not found")
    
    # Check data directory
    config = Config()
    data_dir = config.get_path('data')
    if not list(data_dir.glob('*')):
        logger.warning(f"Data directory is empty: {data_dir}")
        logger.info("Download datasets before training")
    
    logger.info("Remote setup complete!")


if __name__ == '__main__':
    # Test configuration
    config = Config()
    print("Configuration loaded:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Test remote setup
    setup_remote_training()
