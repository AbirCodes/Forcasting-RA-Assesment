@echo off
echo ================================================================================
echo PGCB Forecasting Pipeline - Auto Setup and Run
echo ================================================================================
echo.

echo Step 1: Installing required packages...
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels lightgbm xgboost torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 tensorflow neuralforecast optuna prophet

echo.
echo Step 2: Running forecasting pipeline...
python forecast_pipeline_complete.py

echo.
echo ================================================================================
echo Pipeline execution complete!
echo Check EDA_Output/ for results and visualizations
echo ================================================================================
pause
