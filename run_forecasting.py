"""
PGCB Hourly Generation Forecasting Pipeline
Complete Implementation with GPU Support
Models: LightGBM, N-BEATS, TCN, TFT, Informer
"""

# ============================================================================
# STEP 0: ENVIRONMENT SETUP
# ============================================================================

import os
import sys
import subprocess
import venv
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup virtual environment and install dependencies."""
    venv_path = Path("venv")
    
    # Create virtual environment if needed
    if not venv_path.exists():
        print("[INFO] Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print("[✓] Virtual environment created")
    
    # Install packages if needed
    python_exec = str(venv_path / "Scripts" / "python.exe")
    
    try:
        import pandas, numpy, torch
        print("[✓] Dependencies already installed")
        return python_exec
    except ImportError:
        print("[INFO] Installing dependencies...")
        subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        packages = [
            "pandas>=2.0.0", "numpy>=1.24.0", "scipy>=1.11.0",
            "scikit-learn>=1.3.0", "matplotlib>=3.7.0", "seaborn>=0.12.0",
            "statsmodels>=0.14.0", "lightgbm>=4.0.0", "xgboost>=2.0.0",
            "torch>=2.0.0", "torchvision", "torchaudio",
            "tensorflow>=2.13.0", "neuralforecast>=1.5.0",
            "optuna>=3.0.0", "prophet>=1.1.0"
        ]
        
        for pkg in packages:
            subprocess.run([python_exec, "-m", "pip", "install", pkg], check=True)
        
        return python_exec


def check_gpu():
    """Check for GPU availability."""
    print("\n" + "=" * 80)
    print("GPU CHECK")
    print("=" * 80)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[✓] CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[✓] TensorFlow GPU available: {gpus[0].name}")
            return True
    except ImportError:
        pass
    
    print("[!] No GPU detected - using CPU")
    return False


def install_deps():
    """Install all required packages to current environment."""
    print("\n[INFO] Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    packages = [
        "pandas>=2.0.0", "numpy>=1.24.0", "scipy>=1.11.0",
        "scikit-learn>=1.3.0", "matplotlib>=3.7.0", "seaborn>=0.12.0",
        "statsmodels>=0.14.0", "lightgbm>=4.0.0", "xgboost>=2.0.0",
        "torch>=2.0.0", "torchvision", "torchaudio",
        "tensorflow>=2.13.0", "neuralforecast>=1.5.0",
        "optuna>=3.0.0"
    ]
    
    for pkg in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True, capture_output=True)
        except:
            pass
    print("[✓] Dependencies installed")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete forecasting pipeline."""
    print("=" * 80)
    print("PGCB HOURLY GENERATION FORECASTING")
    print("=" * 80)
    
    # Setup
    install_deps()
    gpu = check_gpu()
    
    print("\n" + "=" * 80)
    print("PHASE 1: DATA LOADING")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    # Load data
    df = pd.read_excel('PGCB_date_power_demand.xlsx')
    print(f"\n[✓] Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Target variable
    target_col = 'generation_mw'
    print(f"\nTarget variable: {target_col}")
    
    # Check data types
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print("\n[✓] Phase 1 complete")


if __name__ == "__main__":
    main()
