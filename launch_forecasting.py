#!/usr/bin/env python
"""
PGCB Forecasting Pipeline Launcher
Auto-creates virtual environment, activates it, installs dependencies, and runs the pipeline
"""

import subprocess
import sys
import os
import venv
from pathlib import Path
import time

def setup_and_run():
    """Create venv, activate, install deps, run pipeline."""
    print("="*70)
    print("PGCB HOURLY GENERATION FORECASTING PIPELINE")
    print("Auto Setup: Venv Creation → Activation → Install → Run")
    print("="*70)
    
    # Step 1: Create virtual environment
    print("\n[STEP 1/4] Creating virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("[✓] Virtual environment already exists")
    else:
        print("[INFO] Creating new virtual environment at ./venv")
        venv.create(venv_path, with_pip=True)
        print("[✓] Virtual environment created")
    
    # Determine Python executable in venv
    if os.name == 'nt':  # Windows
        venv_python = str(venv_path / "Scripts" / "python.exe")
    else:  # Unix/Linux/Mac
        venv_python = str(venv_path / "bin" / "python")
    
    print(f"[INFO] Using: {venv_python}")
    
    # Step 2: Upgrade pip
    print("\n[STEP 2/4] Upgrading pip...")
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("[✓] Pip upgraded")
    except Exception as e:
        print(f"[!] Warning: Could not upgrade pip: {e}")
    
    # Step 3: Install dependencies
    print("\n[STEP 3/4] Installing dependencies...")
    packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "statsmodels>=0.14.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "tensorflow>=2.13.0",
        "neuralforecast>=1.5.0",
        "optuna>=3.0.0",
        "openpyxl"  # For reading Excel files
    ]
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        try:
            result = subprocess.run([venv_python, "-m", "pip", "install", pkg], 
                                  check=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    [✓] {pkg}")
            else:
                print(f"    [!] {pkg} - warning: {result.stderr[:200]}")
        except subprocess.CalledProcessError as e:
            print(f"    [✗] {pkg} - failed: {e}")
    
    print("[✓] Dependencies installed")
    
    # Step 4: Run the pipeline
    print("\n[STEP 4/4] Running forecasting pipeline...")
    print("="*70)
    
    # Run with the venv Python
    result = subprocess.run([venv_python, "forecast_pipeline_complete.py"], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nOutput files:")
        print("  • forecast_results.csv")
        print("  • forecast_*.png")
        print("  • residuals_*.png")
        print("  • feature_importance.csv")
        print("  • scaler.pkl")
    else:
        print("\n" + "="*70)
        print("✗ PIPELINE FAILED")
        print("="*70)
        print("Check error messages above")


if __name__ == "__main__":
    setup_and_run()
