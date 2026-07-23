# Environment Setup Guide

## How the Virtual Environment Works

### Overview
The forecasting pipeline uses a **virtual environment (venv)** to isolate dependencies and avoid conflicts with your system Python installation.

### Setup Process (Automatic)

When you run `python launch_forecasting.py` or `setup_and_run.bat`, the following happens:

1. **Create Virtual Environment**
   - Checks if `venv/` directory exists
   - If not, creates a new virtual environment using `python -m venv venv`
   - Location: `d:\RA Assesment\venv\`

2. **Activate Virtual Environment**
   - **Windows**: `venv\Scripts\activate.bat`
   - **Linux/Mac**: `venv/bin/activate`
   - This ensures all packages install to the venv, not system Python

3. **Upgrade pip**
   - Runs: `pip install --upgrade pip`
   - Ensures latest pip version for package installation

4. **Install Dependencies**
   - Installs all required packages to the venv:
     - Core: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, statsmodels
     - ML: lightgbm, xgboost
     - Deep Learning: torch (with GPU support), tensorflow
     - Forecasting: neuralforecast
     - Utilities: optuna, openpyxl

5. **Run Pipeline**
   - Executes: `venv/bin/python forecast_pipeline_complete.py` (Linux/Mac)
   - Executes: `venv\Scripts\python.exe forecast_pipeline_complete.py` (Windows)
   - Uses the venv's Python, so all installed packages are available

### Manual Setup (If Needed)

If you want to set up manually:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install pandas>=2.0.0 numpy>=1.24.0 scipy>=1.11.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0 statsmodels>=0.14.0 lightgbm>=4.0.0 xgboost>=2.0.0 torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 tensorflow>=2.13.0 neuralforecast>=1.5.0 optuna>=3.0.0 openpyxl

# Run pipeline
python forecast_pipeline_complete.py

# Deactivate when done
deactivate
```

### Directory Structure After Setup

```
d:\RA Assesment\
├── venv/                      # Virtual environment (created automatically)
│   ├── Scripts/              # Windows executables
│   │   ├── python.exe
│   │   ├── activate.bat
│   │   ├── activate.ps1
│   │   └── ...
│   ├── bin/                  # Linux/Mac executables
│   │   ├── python
│   │   ├── activate
│   │   └── ...
│   └── Lib/                  # Python libraries
│       └── site-packages/    # Installed packages
├── forecast_pipeline_complete.py
├── launch_forecasting.py
├── setup_and_run.bat
└── ...
```

### Benefits of Using Virtual Environment

1. **Isolation**: Dependencies don't interfere with system Python
2. **Reproducibility**: Same versions work on any machine
3. **Clean uninstall**: Delete `venv/` folder to remove everything
4. **Version control**: Different projects can have different Python versions
5. **No admin rights needed**: All packages installed locally

### Troubleshooting

**Problem**: "Python is not recognized"
- Solution: Ensure Python is in your PATH, or use full path to Python executable

**Problem**: "Permission denied" during installation
- Solution: Run as administrator or check folder permissions

**Problem**: "pip is not found"
- Solution: Use `python -m pip` instead of `pip`

**Problem**: "CUDA out of memory"
- Solution: Reduce batch size or use CPU mode by removing GPU packages

### Using the Virtual Environment

Once set up, you can use the venv Python directly:

```bash
# Windows
venv\Scripts\python.exe your_script.py

# Linux/Mac
venv/bin/python your_script.py
```

Or activate and use normally:

```bash
# Windows
venv\Scripts\activate
python your_script.py
deactivate

# Linux/Mac
source venv/bin/activate
python your_script.py
deactivate
```

### Cleaning Up

To remove the virtual environment:

```bash
# Windows
rmdir /s /q venv

# Linux/Mac
rm -rf venv
```

This removes all installed packages and the venv directory.
