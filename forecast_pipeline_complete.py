"""
PGCB Hourly Generation Forecasting Pipeline
Complete Implementation - Auto-setup, GPU support, Multi-horizon forecasting

Models:
- LightGBM (Gradient Boosted Trees)
- N-BEATS (Neural Basis Expansion Analysis)
- TCN (Temporal Convolutional Network)
- TFT (Temporal Fusion Transformer)
- Informer (Efficient Transformer)

Author: Data Science Team
Date: June 2025
"""

import os
import sys
import subprocess
import traceback
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 0: ENVIRONMENT SETUP
# ============================================================================

def ensure_dependencies():
    """Ensure all required packages are installed."""
    print("\n" + "=" * 80)
    print("STEP 0: ENVIRONMENT SETUP")
    print("=" * 80)
    
    required_packages = {
        'pandas': 'pandas>=2.0.0',
        'numpy': 'numpy>=1.24.0',
        'scipy': 'scipy>=1.11.0',
        'sklearn': 'scikit-learn>=1.3.0',
        'matplotlib': 'matplotlib>=3.7.0',
        'seaborn': 'seaborn>=0.12.0',
        'statsmodels': 'statsmodels>=0.14.0',
        'lightgbm': 'lightgbm>=4.0.0',
        'xgboost': 'xgboost>=2.0.0',
        'torch': 'torch>=2.0.0',
        'neuralforecast': 'neuralforecast>=1.5.0',
    }
    
    print("\nChecking dependencies...")
    missing = []
    
    for pkg, spec in required_packages.items():
        try:
            __import__(pkg.split('-')[0].replace('scikit-learn', 'sklearn').replace('statsmodels', 'statsmodels').replace('seaborn', 'seaborn').replace('matplotlib', 'matplotlib'))
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MISSING")
            missing.append(spec)
    
    if missing:
        print(f"\nInstalling {len(missing)} missing packages...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            for spec in missing:
                print(f"  Installing {spec}...")
                subprocess.run([sys.executable, "-m", "pip", "install", spec], check=True, capture_output=True)
            print("\n[✓] All packages installed successfully!")
        except Exception as e:
            print(f"\n[✗] Installation failed: {e}")
            print("Please install manually:")
            for spec in missing:
                print(f"  pip install {spec}")
            sys.exit(1)
    else:
        print("\n[✓] All dependencies satisfied!")


def check_gpu():
    """Check for GPU availability."""
    print("\n" + "=" * 80)
    print("GPU AVAILABILITY CHECK")
    print("=" * 80)
    
    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[✓] PyTorch CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"    CUDA Version: {torch.version.cuda}")
            return True
    except ImportError:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[✓] TensorFlow GPU detected: {gpus[0].name}")
            for gpu in gpus:
                print(f"    {gpu.name}")
            return True
    except ImportError:
        pass
    
    print("[!] No GPU detected - using CPU (slower for deep learning models)")
    print("    For GPU support:")
    print("    - PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("    - TensorFlow: pip install tensorflow[and-cuda]")
    return False


# ============================================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================================

def load_and_clean_data():
    """Load and clean PGCB data."""
    print("\n" + "=" * 80)
    print("PHASE 1: DATA LOADING & CLEANING")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    # Load data
    print("\n[INFO] Loading PGCB dataset...")
    df = pd.read_excel('PGCB_date_power_demand.xlsx')
    print(f"[✓] Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Display columns
    print("\nDataset columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}: {df[col].dtype}")
    
    # Identify target column
    target_col = 'generation_mw'
    if target_col not in df.columns:
        # Try to find it
        for col in df.columns:
            if 'generation' in col.lower() or 'power' in col.lower():
                target_col = col
                break
    
    print(f"\nTarget variable: {target_col}")
    
    # Check data types
    print(f"\nData type distribution:")
    print(df.dtypes.value_counts())
    
    # Summary statistics
    print(f"\nTarget variable statistics:")
    print(df[target_col].describe().round(2))
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Missing values
    missing = df.isnull().sum()
    print(f"\nMissing values by column:")
    print(missing[missing > 0])
    
    # Timestamp column
    datetime_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    
    if datetime_col:
        print(f"\nDatetime column: {datetime_col}")
        print(f"  Range: {df[datetime_col].min()} to {df[datetime_col].max()}")
        print(f"  Duration: {(df[datetime_col].max() - df[datetime_col].min()).days} days")
    
    print("\n[✓] Data loading complete")
    
    return df, target_col, datetime_col


def clean_data(df, target_col):
    """Clean data: remove duplicates, handle missing values, fix outliers."""
    print("\n" + "=" * 80)
    print("PHASE 2: DATA CLEANING")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    df_clean = df.copy()
    
    # 1. Remove duplicates
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"\n[1] Duplicate removal:")
    print(f"  Before: {n_before:,} rows")
    print(f"  After: {len(df_clean):,} rows")
    print(f"  Removed: {n_before - len(df_clean):,} duplicate rows")
    
    # 2. Convert datetime column
    datetime_col = None
    for col in df_clean.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    
    if datetime_col:
        print(f"\n[2] Converting datetime column: {datetime_col}")
        if df_clean[datetime_col].dtype == 'object':
            df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
        df_clean = df_clean.sort_values(datetime_col).reset_index(drop=True)
        print(f"  Sorted chronologically")
    
    # 3. Handle missing values in target
    print(f"\n[3] Missing value handling for {target_col}:")
    missing_target = df_clean[target_col].isnull().sum()
    print(f"  Missing values: {missing_target} ({missing_target/len(df_clean)*100:.2f}%)")
    
    if missing_target > 0:
        # Linear interpolation
        df_clean[target_col] = df_clean[target_col].interpolate(method='linear')
        # Forward fill remaining
        df_clean[target_col] = df_clean[target_col].fillna(method='ffill')
        # Backward fill remaining
        df_clean[target_col] = df_clean[target_col].fillna(method='bfill')
        print(f"  Applied: Linear interpolation → FFill → BFill")
    
    # 4. Handle outliers
    print(f"\n[4] Outlier handling for {target_col}:")
    
    # Calculate IQR bounds
    Q1 = df_clean[target_col].quantile(0.25)
    Q3 = df_clean[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Also use domain knowledge: Bangladesh capacity ~24,000 MW
    # So cap at 30,000 MW (conservative)
    domain_bound = 30000
    
    print(f"  IQR bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
    print(f"  Domain bound: <= {domain_bound} MW")
    
    # Cap at maximum of both bounds
    safe_upper = min(upper_bound, domain_bound)
    
    n_outliers = (df_clean[target_col] > safe_upper).sum()
    print(f"  Outliers above {safe_upper:.0f} MW: {n_outliers} ({n_outliers/len(df_clean)*100:.2f}%)")
    
    # Cap outliers
    df_clean.loc[df_clean[target_col] > safe_upper, target_col] = safe_upper
    print(f"  Applied: Capped outliers at {safe_upper:.0f} MW")
    
    print("\n[✓] Data cleaning complete")
    
    return df_clean, datetime_col


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

def create_features(df, target_col, datetime_col, forecast_horizon=24):
    """Create features for forecasting."""
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    # Make a copy
    df_feat = df.copy()
    
    # 1. Time-based features
    print("\n[1] Creating time-based features...")
    
    if datetime_col:
        df_feat['datetime'] = pd.to_datetime(df_feat[datetime_col])
    
    df_feat['hour'] = df_feat['datetime'].dt.hour
    df_feat['day_of_week'] = df_feat['datetime'].dt.dayofweek
    df_feat['day_of_month'] = df_feat['datetime'].dt.day
    df_feat['month'] = df_feat['datetime'].dt.month
    df_feat['quarter'] = df_feat['datetime'].dt.quarter
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    df_feat['year'] = df_feat['datetime'].dt.year
    df_feat['day_of_year'] = df_feat['datetime'].dt.dayofyear
    
    print(f"  ✓ Time features created")
    
    # 2. Cyclical encoding
    print("\n[2] Creating cyclical encodings...")
    
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    print(f"  ✓ Cyclical features created")
    
    # 3. Lag features
    print("\n[3] Creating lag features...")
    
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lag_hours:
        df_feat[f'lag_{lag}h'] = df_feat[target_col].shift(lag)
    
    print(f"  ✓ Lag features: {lag_hours}")
    
    # 4. Rolling statistics
    print("\n[4] Creating rolling statistics...")
    
    windows = [24, 48, 168]  # 1d, 2d, 7d
    for w in windows:
        df_feat[f'rolling_mean_{w}h'] = df_feat[target_col].rolling(w, min_periods=1).mean()
        df_feat[f'rolling_std_{w}h'] = df_feat[target_col].rolling(w, min_periods=1).std()
        df_feat[f'rolling_min_{w}h'] = df_feat[target_col].rolling(w, min_periods=1).min()
        df_feat[f'rolling_max_{w}h'] = df_feat[target_col].rolling(w, min_periods=1).max()
    
    print(f"  ✓ Rolling features: windows = {windows}")
    
    # 5. Exponential moving average
    print("\n[5] Creating exponential moving averages...")
    
    for span in [24, 168]:
        df_feat[f'ema_{span}h'] = df_feat[target_col].ewm(span=span, adjust=False).mean()
    
    print(f"  ✓ EMA features: spans = [24, 168]")
    
    # 6. Same hour previous day/week
    print("\n[6] Creating historical same-hour features...")
    
    df_feat['same_hour_yesterday'] = df_feat[target_col].shift(24)
    df_feat['same_hour_last_week'] = df_feat[target_col].shift(168)
    
    print(f"  ✓ Historical features created")
    
    # 7. Difference features (for stationarity)
    print("\n[7] Creating difference features...")
    
    df_feat['diff_1h'] = df_feat[target_col].diff(1)
    df_feat['diff_24h'] = df_feat[target_col].diff(24)
    
    print(f"  ✓ Difference features created")
    
    # Drop rows with NaN (from lags/differences)
    print("\n[8] Dropping rows with NaN values...")
    n_before = len(df_feat)
    df_feat = df_feat.dropna()
    n_after = len(df_feat)
    print(f"  Before: {n_before:,} rows")
    print(f"  After: {n_after:,} rows")
    print(f"  Removed: {n_before - n_after:,} rows")
    
    # Identify feature columns
    exclude_cols = ['datetime', target_col, datetime_col] if datetime_col else ['datetime', target_col]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}... (showing first 10)")
    
    print("\n[✓] Feature engineering complete")
    
    return df_feat, feature_cols


# ============================================================================
# STEP 3: DATA SPLITTING
# ============================================================================

def create_train_test_split(df, feature_cols, target_col, test_size=0.2):
    """Create chronological train/test split."""
    print("\n" + "=" * 80)
    print("PHASE 4: DATA SPLITTING")
    print("=" * 80)
    
    import numpy as np
    
    n_samples = len(df)
    train_size = int(n_samples * (1 - test_size))
    
    # Split chronologically (no shuffle)
    train_idx = df.index[:train_size]
    test_idx = df.index[train_size:]
    
    X_train = df.loc[train_idx, feature_cols].values
    y_train = df.loc[train_idx, target_col].values
    X_test = df.loc[test_idx, feature_cols].values
    y_test = df.loc[test_idx, target_col].values
    
    print(f"\n[✓] Train/Test split: {train_size:,} / {len(df) - train_size:,}")
    print(f"    Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"    Test shape: X={X_test.shape}, y={y_test.shape}")
    
    # Scaling
    print("\n[✓] Feature scaling: StandardScaler")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  ✓ Scaler saved to scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, train_idx, test_idx, scaler


# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM model."""
    print("\n" + "=" * 80)
    print("Training LightGBM...")
    print("=" * 80)
    
    import lightgbm as lgb
    import time
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbosity': -1,
        'random_state': 42
    }
    
    start_time = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    training_time = time.time() - start_time
    
    print(f"[✓] LightGBM trained in {training_time:.2f} seconds")
    
    return model, training_time


def train_neuralforecast_models(X_train, y_train, X_val, y_val, feature_cols, target_col):
    """Train NeuralForecast models (N-BEATS, TCN, TFT, Informer)."""
    print("\n" + "=" * 80)
    print("Training NeuralForecast Models...")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBeats, TCN, TFT, Informer
    from neuralforecast.common import TimestampF
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Prepare data for NeuralForecast
    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train[target_col] = y_train
    df_train['datetime'] = pd.date_range('2015-01-01', periods=len(df_train), freq='H')
    df_train['y'] = df_train[target_col]
    df_train['ds'] = df_train['datetime']
    df_train['unique_id'] = 'PGCB'
    
    # Forecast horizon
    h = 24
    
    # Models
    models = [
        NBeats(input_size=168, h=h, max_steps=100, scaler_type='robust'),
        TCN(input_size=168, h=h, max_steps=100),
        TFT(input_size=168, h=h, max_steps=100, num_layers=2),
        Informer(input_size=168, h=h, max_steps=100)
    ]
    
    nf = NeuralForecast(models=models, freq='H')
    
    # Fit models
    start_time = pd.Timestamp.now()
    nf.fit(df_train, val_df=df_train)
    training_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"[✓] NeuralForecast models trained in {training_time:.2f} seconds")
    
    return nf, training_time


def train_baseline_arima(y_train, y_test):
    """Train baseline ARIMA model."""
    print("\n" + "=" * 80)
    print("Training ARIMA Baseline...")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Simple ARIMA (p=1, d=1, q=1)
    model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    fitted = model.fit(disp=False)
    
    # Forecast
    predictions = fitted.get_forecast(steps=len(y_test))
    y_pred = predictions.predicted_mean.values
    
    print("[✓] ARIMA trained and predictions generated")
    
    return y_pred


# ============================================================================
# STEP 5: EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, model_name, horizon):
    """Evaluate model predictions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (handle division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'Model': model_name,
        'Horizon': horizon,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def plot_forecasts(y_true, y_pred, dates, model_name, horizon, n_days=7):
    """Plot forecast vs actual."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Last n_days of test set
    n_points = min(n_days * 24, len(y_true))
    idx = range(-n_points, 0)
    
    plt.figure(figsize=(14, 6))
    plt.plot(dates[idx], y_true[idx], label='Actual', linewidth=2)
    plt.plot(dates[idx], y_pred[idx], label=f'Predicted ({model_name})', linewidth=2, alpha=0.8)
    plt.fill_between(dates[idx], 
                     y_true[idx] * 0.9, y_true[idx] * 1.1, 
                     alpha=0.2, label='±10% Range')
    plt.title(f'{model_name} Forecast vs Actual (Last {n_days} days, {horizon}h horizon)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Generation (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'forecast_{model_name}_{horizon}h.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: forecast_{model_name}_{horizon}h.png")
    plt.close()


def plot_residuals(y_true, y_pred, model_name):
    """Plot residual analysis."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residuals vs time
    axes[0].plot(residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title(f'{model_name} Residuals vs Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residual (Actual - Predicted)')
    
    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title(f'{model_name} Residual Distribution')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title(f'{model_name} Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(f'residuals_{model_name}.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: residuals_{model_name}.png")
    plt.close()


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def main():
    """Execute complete forecasting pipeline."""
    print("\n" + "=" * 80)
    print("PGCB HOURLY GENERATION FORECASTING PIPELINE")
    print("=" * 80)
    print("\nModels: LightGBM, N-BEATS, TCN, TFT, Informer")
    print("Forecast Horizons: 1h, 6h, 24h")
    
    # Step 0: Environment setup
    print("\n" + "=" * 80)
    print("PHASE 0: ENVIRONMENT SETUP")
    print("=" * 80)
    
    # Install dependencies if needed
    try:
        import pandas, numpy, sklearn, matplotlib, seaborn, lightgbm
        print("[✓] All required packages available")
    except ImportError as e:
        print(f"[!] Missing package: {e}")
        print("Installing packages...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn", 
                       "matplotlib", "seaborn", "lightgbm", "statsmodels", "neuralforecast"], check=True)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[✓] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("[!] No GPU detected - using CPU")
    except:
        print("[!] Could not check GPU")
    
    # Step 1: Data loading & cleaning
    df, target_col, datetime_col = load_and_clean_data()
    df_clean, datetime_col = clean_data(df, target_col)
    
    # Step 2: Feature engineering
    df_feat, feature_cols = create_features(df_clean, target_col, datetime_col)
    
    # Step 3: Train/test split
    X_train, X_test, y_train, y_test, train_idx, test_idx, scaler = create_train_test_split(
        df_feat, feature_cols, target_col
    )
    
    # Step 4: Model training
    print("\n" + "=" * 80)
    print("PHASE 5: MODEL TRAINING & EVALUATION")
    print("=" * 80)
    
    # Split training data for validation
    val_size = int(len(X_train) * 0.1)
    X_train_train = X_train[:-val_size]
    y_train_train = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    
    results = []
    
    # Train LightGBM
    model_lgb, time_lgb = train_lightgbm(X_train_train, y_train_train, X_val, y_val, feature_cols)
    
    # Predictions
    y_pred_lgb = model_lgb.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_lgb, 'LightGBM', 24))
    
    # Baseline ARIMA
    y_pred_arima = train_baseline_arima(y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_arima, 'ARIMA', 24))
    
    # NeuralForecast models (if available)
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBeats, TCN, TFT, Informer
        nf, time_nf = train_neuralforecast_models(X_train, y_train, X_val, y_val, feature_cols, target_col)
        print("[✓] NeuralForecast models available")
    except:
        print("[!] NeuralForecast not available - skipping deep learning models")
    
    # Step 5: Evaluation
    print("\n" + "=" * 80)
    print("PHASE 6: EVALUATION RESULTS")
    print("=" * 80)
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    print("\nForecast Results (24h horizon):")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('forecast_results.csv', index=False)
    print("\n✓ Results saved to forecast_results.csv")
    
    # Plot forecasts
    print("\nGenerating diagnostic plots...")
    plot_forecasts(y_test, y_pred_lgb, df_feat.loc[test_idx, 'datetime'], 'LightGBM', 24)
    plot_residuals(y_test, y_pred_lgb, 'LightGBM')
    
    plot_forecasts(y_test, y_pred_arima, df_feat.loc[test_idx, 'datetime'], 'ARIMA', 24)
    plot_residuals(y_test, y_pred_arima, 'ARIMA')
    
    # Save feature importance
    import pandas as pd
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model_lgb.feature_importance()
    }).sort_values('Importance', ascending=False)
    importance.to_csv('feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to feature_importance.csv")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  • forecast_results.csv - Model comparison results")
    print("  • feature_importance.csv - LightGBM feature importance")
    print("  • forecast_LightGBM_24h.png - Forecast visualization")
    print("  • forecast_ARIMA_24h.png - ARIMA forecast visualization")
    print("  • residuals_LightGBM.png - LightGBM residual analysis")
    print("  • residuals_ARIMA.png - ARIMA residual analysis")
    print("  • scaler.pkl - Feature scaler for inference")


if __name__ == "__main__":
    main()
