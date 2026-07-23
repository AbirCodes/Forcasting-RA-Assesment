#!/usr/bin/env python
"""
PGCB Forecasting Pipeline Launcher
Auto-installs dependencies and runs the complete pipeline
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completed")
        if result.stdout:
            print(result.stdout[:500])  # First 500 chars
    else:
        print(f"✗ {description} failed")
        if result.stderr:
            print(result.stderr[:500])
        return False
    return True

def main():
    print("="*60)
    print("PGCB HOURLY GENERATION FORECASTING PIPELINE")
    print("Auto Setup & Launch")
    print("="*60)
    
    # Install packages
    print("\nInstalling packages...")
    packages = [
        "pandas>=2.0.0", "numpy>=1.24.0", "scipy>=1.11.0",
        "scikit-learn>=1.3.0", "matplotlib>=3.7.0", "seaborn>=0.12.0",
        "statsmodels>=0.14.0", "lightgbm>=4.0.0", "xgboost>=2.0.0",
        "torch>=2.0.0", "torchvision", "torchaudio",
        "tensorflow>=2.13.0", "neuralforecast>=1.5.0", "optuna>=3.0.0"
    ]
    
    # Use conda if available, otherwise pip
    use_conda = False
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
        use_conda = True
    except:
        pass
    
    if use_conda:
        print("Using conda for package installation")
        subprocess.run(["conda", "install", "-y"] + packages, check=True)
    else:
        print("Using pip for package installation")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        for pkg in packages:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
    
    # Run the pipeline
    print("\n" + "="*60)
    print("Running forecasting pipeline...")
    print("="*60)
    
    result = subprocess.run([sys.executable, "forecast_pipeline_complete.py"], shell=True)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the following files for results:")
        print("  • forecast_results.csv - Model comparison")
        print("  • forecast_*.png - Forecast visualizations")
        print("  • residuals_*.png - Residual analysis")
        print("  • feature_importance.csv - LightGBM importance")
    else:
        print("\n" + "="*60)
        print("✗ PIPELINE FAILED")
        print("="*60)
        print("Check error messages above")

if __name__ == "__main__":
    main()
