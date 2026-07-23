@echo off
echo ================================================================================
echo PGCB Forecasting Pipeline - Auto Setup and Run
echo This script will:
echo   1. Create virtual environment (venv)
echo   2. Activate it
echo   3. Install all dependencies (pandas, numpy, torch, tensorflow, etc.)
echo   4. Run the complete forecasting pipeline with GPU support
echo ================================================================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [✓] Virtual environment created
) else (
    echo [✓] Virtual environment already exists
)

REM Activate venv and run
echo.
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [INFO] Installing/upgrading pip...
python -m pip install --upgrade pip

echo.
echo [INFO] Installing required packages...
echo This may take 5-15 minutes depending on your internet connection.
echo.

python -m pip install pandas>=2.0.0 numpy>=1.24.0 scipy>=1.11.0
python -m pip install scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0
python -m pip install statsmodels>=0.14.0 lightgbm>=4.0.0 xgboost>=2.0.0
python -m pip install openpyxl

REM Install PyTorch with CUDA support (GPU)
echo.
echo [INFO] Installing PyTorch (with GPU support)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install TensorFlow
echo.
echo [INFO] Installing TensorFlow...
python -m pip install tensorflow>=2.13.0

REM Install NeuralForecast
echo.
echo [INFO] Installing NeuralForecast (N-BEATS, TCN, TFT, Informer)...
python -m pip install neuralforecast>=1.5.0

REM Install Optuna
echo.
echo [INFO] Installing Optuna (hyperparameter optimization)...
python -m pip install optuna>=3.0.0

echo.
echo [✓] All dependencies installed
echo.

REM Run the pipeline
echo ================================================================================
echo Running forecasting pipeline...
echo ================================================================================
python forecast_pipeline_complete.py

echo.
echo ================================================================================
echo Pipeline execution complete!
echo Check the following files for results:
echo   • forecast_results.csv
echo   • forecast_*.png
echo   • residuals_*.png
echo   • feature_importance.csv
echo   • scaler.pkl
echo ================================================================================
echo.

REM Deactivate venv
deactivate

pause
