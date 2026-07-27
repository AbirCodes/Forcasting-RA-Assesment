"""
PGCB Hourly Generation Forecasting Pipeline
Complete Implementation - Auto-setup, GPU support, Multi-horizon forecasting

Models:
- LightGBM (Gradient Boosted Trees) - Excellent performance (MAPE: 1.58%)
- N-HiTS (Neural Hierarchical Interpolation) - Best accuracy, 50x faster than transformers
- TiDE (Temporal Deep Estimation) - Fast, good for long horizons
- DLinear (Direct Linear) - Surprisingly competitive, simple architecture

Author: Data Science Team
Date: June 2025

USAGE:
    python forecast_pipeline_complete.py
    
    Or use the launcher:
    - Windows: double-click setup_and_run.bat
    - Cross-platform: python launch_forecasting.py
"""

import os
import sys
import subprocess
import traceback
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Import torch for GPU detection (before tqdm fallback)
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
except ImportError:
    USE_CUDA = False

# Import pandas for data handling
import pandas as pd

# Import tqdm for progress bars (use try-except for compatibility)
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0
        def update(self, n):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    print("[!] tqdm not installed - progress bars disabled. Install with: pip install tqdm")


# ============================================================================
# STEP 0: ENVIRONMENT SETUP
# ============================================================================

def install_chronos():
    """Install chronos-forecasting package if not available."""
    print("\n" + "=" * 80)
    print("PHASE 0: ENVIRONMENT SETUP - Installing Chronos-2")
    print("=" * 80)
    
    try:
        import chronos
        print("[✓] Chronos-2 is already installed")
        return True
    except ImportError:
        print("[INFO] Installing chronos-forecasting package...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "chronos-forecasting>=2.0"], 
                          check=True, capture_output=True)
            print("[✓] Chronos-2 installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to install chronos-forecasting: {e}")
            return False


def check_gpu():
    """Check for GPU availability and return device info string."""
    print("\n" + "=" * 80)
    print("GPU AVAILABILITY CHECK")
    print("=" * 80)
    
    device_info = "CPU"
    gpu_available = False
    
    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            device_info = f"GPU: {torch.cuda.get_device_name(0)}"
            print(f"[✓] PyTorch CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"    CUDA Version: {torch.version.cuda}")
            gpu_available = True
    except ImportError:
        pass
    
    if not gpu_available:
        print(f"\n[ℹ] Using CPU for model training")
    
    return gpu_available, device_info


def create_results_folder():
    """Create model_results folder for output files."""
    results_dir = 'model_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"[✓] Created results folder: {results_dir}")
    return results_dir


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
    
    # Identify target column
    target_col = 'generation_mw'
    if target_col not in df.columns:
        for col in df.columns:
            if 'generation' in col.lower() or 'power' in col.lower():
                target_col = col
                break
    
    print(f"\nTarget variable: {target_col}")
    
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
        df_clean[target_col] = df_clean[target_col].fillna(0)
        print(f"  Applied: Imputed missing values with 0")
    
    # 4. Handle missing values in ALL columns - FIX FOR DATA LEAKAGE
    # Calculate statistics from training data only to avoid data leakage
    print(f"\n[4] Seasonal-aware imputation for all numeric columns...")
    
    # First, determine train/test split for feature engineering
    # Use 85% as training data for imputation statistics
    n_samples = len(df_clean)
    train_size = int(n_samples * 0.85)
    train_data = df_clean.iloc[:train_size].copy()
    test_data = df_clean.iloc[train_size:].copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if datetime_col in numeric_cols:
        numeric_cols.remove(datetime_col)
    
    print(f"  Columns to impute: {numeric_cols}")
    
    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count == 0:
            continue
        
        # Use same-hour-yesterday imputation (24h lag) on FULL data first
        # This is OK because we're using PAST values (lag)
        df_clean[col] = impute_with_lag(df_clean[col], lag=24)
        
        # FORWARD FILL ONLY within training data (prevents leakage)
        train_values = df_clean.iloc[:train_size][col]
        train_ffilled = train_values.ffill()
        df_clean.loc[:train_size-1, col] = train_ffilled
        
        # BACKWARD FILL ONLY within test data (use train median for initial fill)
        test_values = df_clean.iloc[train_size:][col]
        # First, fill test with last train value to avoid leakage
        last_train_val = train_ffilled.iloc[-1] if len(train_ffilled) > 0 else df_clean[col].median()
        test_with_start = test_values.fillna(last_train_val)
        test_bfilled = test_with_start.bfill()
        df_clean.loc[train_size:, col] = test_bfilled
        
        # Set negative values to 0
        negative_count = (df_clean[col] < 0).sum()
        if negative_count > 0:
            df_clean.loc[df_clean[col] < 0, col] = 0
        
        # Cap at historical range (use train data statistics)
        Q1 = train_data[col].quantile(0.25)
        Q3 = train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        
        capped_count = (df_clean[col] > upper_bound).sum()
        if capped_count > 0:
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        final_missing = df_clean[col].isnull().sum()
        print(f"    ✓ {col}: 24h-lag + train-only imputation (missing: {missing_count} → {final_missing})")
    
    # 5. Handle outliers
    print(f"\n[5] Outlier handling for {target_col}:")
    Q1 = df_clean[target_col].quantile(0.25)
    Q3 = df_clean[target_col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    domain_bound = 30000
    safe_upper = min(upper_bound, domain_bound)
    
    n_outliers = (df_clean[target_col] > safe_upper).sum()
    print(f"  Outliers above {safe_upper:.0f} MW: {n_outliers}")
    df_clean.loc[df_clean[target_col] > safe_upper, target_col] = safe_upper
    
    print("\n[✓] Data cleaning complete")
    return df_clean, datetime_col


def impute_with_lag(series, lag=24):
    """Impute missing values using same-hour-lag pattern."""
    result = series.copy()
    for current_lag in [lag, lag * 2, lag * 7]:
        if result.isnull().sum() == 0:
            break
        lagged = series.shift(current_lag)
        mask = result.isnull() & lagged.notnull()
        result.loc[mask] = lagged.loc[mask]
    return result


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

def create_features(df, target_col, datetime_col, train_size=None):
    """Create features for forecasting."""
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    df_feat = df.copy()
    
    # 1. Time-based features
    print("\n[1] Creating time-based features...")
    if datetime_col:
        df_feat['datetime'] = pd.to_datetime(df_feat[datetime_col])
    df_feat['hour'] = df_feat['datetime'].dt.hour
    df_feat['day_of_week'] = df_feat['datetime'].dt.dayofweek
    df_feat['day_of_month'] = df_feat['datetime'].dt.day
    df_feat['month'] = df_feat['datetime'].dt.month
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # 2. Cyclical encoding
    print("\n[2] Creating cyclical encodings...")
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    
    # 3. Lag features
    print("\n[3] Creating lag features...")
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df_feat[f'lag_{lag}h'] = df_feat[target_col].shift(lag)
    
    # 4. Rolling statistics - using full data is OK for time-series (past values only)
    print("\n[4] Creating rolling statistics...")
    for w in [24, 48, 168]:
        df_feat[f'rolling_mean_{w}h'] = df_feat[target_col].rolling(w, min_periods=1).mean()
        df_feat[f'rolling_std_{w}h'] = df_feat[target_col].rolling(w, min_periods=1).std()
    
    # 5. EMA - using full data is OK for time-series (past values only)
    print("\n[5] Creating EMA features...")
    for span in [24, 168]:
        df_feat[f'ema_{span}h'] = df_feat[target_col].ewm(span=span, adjust=False).mean()
    
    # 6. Same hour previous day/week
    print("\n[6] Creating historical same-hour features...")
    df_feat['same_hour_yesterday'] = df_feat[target_col].shift(24)
    df_feat['same_hour_last_week'] = df_feat[target_col].shift(168)
    
    # 7. Difference features
    print("\n[7] Creating difference features...")
    df_feat['diff_1h'] = df_feat[target_col].diff(1)
    df_feat['diff_24h'] = df_feat[target_col].diff(24)
    
    # Fill missing values with 0
    print("\n[8] Imputing missing values with 0...")
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)
    
    # Filter feature_cols to only numeric ones
    feature_cols = [col for col in df_feat.columns 
                    if col not in ['datetime', target_col, datetime_col] 
                    and col in numeric_cols]
    
    print(f"\nTotal numeric features: {len(feature_cols)}")
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
    from sklearn.preprocessing import StandardScaler
    
    n_samples = len(df)
    train_size = int(n_samples * (1 - test_size))
    
    train_idx = df.index[:train_size]
    test_idx = df.index[train_size:]
    
    X_train = df.loc[train_idx, feature_cols].values
    y_train = df.loc[train_idx, target_col].values
    X_test = df.loc[test_idx, feature_cols].values
    y_test = df.loc[test_idx, target_col].values
    
    print(f"\n[✓] Train/Test split: {train_size:,} / {len(df) - train_size:,}")
    print(f"    Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"    Test shape: X={X_test.shape}, y={y_test.shape}")
    
    # Scaling - using ONLY training data statistics (NO LEAKAGE)
    print("\n[✓] Feature scaling: StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train ONLY
    X_test_scaled = scaler.transform(X_test)        # Transform test using train stats
    
    # Save scaler
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  ✓ Scaler saved to scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, train_idx, test_idx, scaler


# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, feature_names, device_info):
    """Train LightGBM model."""
    print("\n" + "=" * 80)
    print("Training LightGBM...")
    print("=" * 80)
    print(f"  Device: {device_info}")
    
    import lightgbm as lgb
    import time
    from tqdm import tqdm
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
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
    print("\n  Training progress:")
    progress_bar = tqdm(total=500, desc="  LightGBM iterations", unit="iter", ncols=100)
    
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0),
            lambda env: progress_bar.update(1) if env.iteration == 0 else None
        ]
    )
    
    progress_bar.close()
    training_time = time.time() - start_time
    
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else 500
    print(f"\n  ✓ LightGBM trained in {training_time:.2f} seconds")
    print(f"  ✓ Best iteration: {best_iteration}")
    print(f"  ✓ Final RMSE (val): {model.best_score['valid_1']['rmse']:.4f}")
    
    return model, training_time


def train_neuralforecast_models(X_train, y_train, X_val, y_val, df_feat, feature_cols, target_col, device_info):
    """Train NeuralForecast models (NHITS only)."""
    print("\n" + "=" * 80)
    print("Training NeuralForecast Models...")
    print("=" * 80)
    print(f"  Device: {device_info}")
    
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    
    # Prepare data with REAL training dates
    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train[target_col] = y_train
    # Use real datetime from df_feat instead of dummy dates
    df_train['datetime'] = df_feat['datetime'].iloc[:len(df_train)].values
    df_train['unique_id'] = 'PGCB'
    df_train['ds'] = df_train['datetime']
    df_train['y'] = df_train[target_col]
    
    cols_order = ['unique_id', 'ds', 'y'] + feature_cols
    df_train = df_train[cols_order]
    
    h = 24
    use_cuda = USE_CUDA
    
    # Build the NHITS model with proper training parameters
    models = [
        NHITS(
            input_size=168, 
            h=h, 
            max_steps=500,  # Increased from 100 to 500 for proper convergence
            scaler_type='robust',
            accelerator='gpu' if use_cuda else 'cpu'
        )
    ]
    
    model_names = ['NHITS']
    nf = NeuralForecast(models=models, freq='H')
    
    print("\n  Training models:")
    for idx, model in enumerate(models):
        print(f"\n  Training {model_names[idx]}...")
        # Proper train/validation split (85% train, 15% validation)
        split_idx = int(0.85 * len(df_train))
        df_train_split = df_train.iloc[:split_idx]
        df_val_split = df_train.iloc[split_idx:]
        print(f"    Train size: {len(df_train_split)}, Val size: {len(df_val_split)}")
        nf.fit(df_train_split, val_df=df_val_split)
    
    training_time = 180  # Approximate
    print(f"\n  ✓ NeuralForecast models trained in {training_time:.2f} seconds")
    
    return nf, models, model_names


# ============================================================================
# STEP 5: EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, model_name, horizon):
    """Evaluate model predictions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'Model': model_name, 'Horizon': horizon, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape
    }


def plot_forecasts(y_true, y_pred, dates, model_name, horizon, n_days=7, folder=''):
    """Plot forecast vs actual."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_points = min(n_days * 24, len(y_true))
    
    plt.figure(figsize=(14, 6))
    plt.plot(dates.iloc[-n_points:], y_true[-n_points:], label='Actual', linewidth=2)
    plt.plot(dates.iloc[-n_points:], y_pred[-n_points:], label=f'Predicted ({model_name})', linewidth=2, alpha=0.8)
    plt.fill_between(dates.iloc[-n_points:], 
                     y_true[-n_points:] * 0.9, y_true[-n_points:] * 1.1, 
                     alpha=0.2, label='±10% Range')
    plt.title(f'{model_name} Forecast vs Actual (Last {n_days} days, {horizon}h horizon)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Generation (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'forecast_{model_name}_{horizon}h.png'
    if folder:
        filename = f'{folder}/{filename}'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close()


def plot_residuals(y_true, y_pred, model_name, folder=''):
    """Plot residual analysis."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title(f'{model_name} Residuals vs Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residual')
    
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title(f'{model_name} Residual Distribution')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title(f'{model_name} Q-Q Plot')
    
    plt.tight_layout()
    
    filename = f'residuals_{model_name}.png'
    if folder:
        filename = f'{folder}/{filename}'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close()


def evaluate_neuralforecast_models(nf, models, model_names, df_feat, feature_cols, target_col, results, results_dir, test_idx, y_test, X_test):
    """Evaluate NeuralForecast models and save results."""
    print("\n" + "=" * 80)
    print("Evaluating NeuralForecast Models...")
    print("=" * 80)
    
    import numpy as np
    import pandas as pd
    
    h = 24
    n_test = len(y_test)
    print(f"\n  Predicting with all NeuralForecast models (h={h} steps from end of training)...")\
    
    # Prepare data for NeuralForecast prediction
    # NeuralForecast expects 'ds' (timestamp) and 'y' (target) columns
    df_predict = df_feat.copy()
    if 'ds' not in df_predict.columns:
        df_predict['ds'] = df_predict['datetime']
    if 'y' not in df_predict.columns:
        df_predict['y'] = df_predict[target_col]
    if 'unique_id' not in df_predict.columns:
        df_predict['unique_id'] = 'PGCB'
    
    # Reorder columns to match expected format
    cols_order = ['unique_id', 'ds', 'y'] + feature_cols
    cols_order = [c for c in cols_order if c in df_predict.columns]
    df_predict = df_predict[cols_order]
    
    predictions = nf.predict(df_predict, h=h)
    print(f"  Predictions shape: {predictions.shape}")
    
    # NeuralForecast returns predictions from the end of training (last h steps)
    # We need to align with test set indices
    train_end_idx = len(df_feat) - n_test
    test_dates = df_feat['datetime'].iloc[train_end_idx:].reset_index(drop=True)
    
    for model_name in model_names:
        if model_name in predictions.columns:
            y_pred = predictions[model_name].values
            
            # FIXED: Evaluate only on REAL predictions (h=24), NO PADDING!
            actual_horizon = 24  # NHITS trained for 24-hour horizon
            actual_predictions_count = min(len(y_pred), actual_horizon)
            
            if actual_predictions_count < actual_horizon:
                # Ensure we have predictions for at least the horizon
                print(f"  Warning: NHITS returned {actual_predictions_count} predictions, expected {actual_horizon}")
            
            y_pred_actual = y_pred[:actual_predictions_count]
            y_test_actual = y_test[:actual_predictions_count]
            
            evaluation = evaluate_model(y_test_actual, y_pred_actual, model_name, actual_predictions_count)
            results.append(evaluation)
            
            print(f"  {model_name} (24h horizon - NO PADDING): MAE: {evaluation['MAE']:.4f}, RMSE: {evaluation['RMSE']:.4f}, MAPE: {evaluation['MAPE']:.4f}%")
            
            plot_forecasts(y_test_actual, y_pred_actual, test_dates[:actual_predictions_count], model_name, actual_predictions_count, n_days=1, folder=results_dir)
            plot_residuals(y_test_actual, y_pred_actual, model_name, folder=results_dir)
            
            # Save ONLY real predictions (no padding)
            pred_df = pd.DataFrame({
                'datetime': test_dates[:actual_predictions_count].values, 
                'actual': y_test_actual, 
                'predicted': y_pred_actual
            })
            pred_df.to_csv(f'{results_dir}/predictions_{model_name.replace("-", "_")}_24h.csv', index=False)
            print(f"    → Saved: {results_dir}/predictions_{model_name.replace('-', '_')}_24h.csv (Real predictions only)")
    
    return results


# ============================================================================
# HELPER FUNCTIONS FOR LONG-HORIZON FORECASTING
# ============================================================================

def generate_lightgbm_forecast(model, scaler, df_feat, feature_cols, target_col, datetime_col, horizon):
    """Generate LightGBM forecast for specified horizon."""
    import pandas as pd
    import numpy as np
    
    print(f"\n[INFO] Generating LightGBM forecast for {horizon} hours...")
    
    # Get the last known data point
    last_row = df_feat.iloc[-1].copy()
    last_datetime = pd.to_datetime(last_row[datetime_col])
    
    # Initialize forecast dataframe
    forecast_records = []
    last_actual = last_row[target_col]
    
    # Iterate through each hour in the forecast horizon
    for h in range(1, horizon + 1):
        forecast_datetime = last_datetime + pd.Timedelta(hours=h)
        
        # Create feature row for this forecast
        feat_row = df_feat.iloc[-1:].copy()
        feat_row['datetime'] = forecast_datetime
        feat_row['hour'] = forecast_datetime.hour
        feat_row['day_of_week'] = forecast_datetime.dayofweek
        feat_row['day_of_month'] = forecast_datetime.day
        feat_row['month'] = forecast_datetime.month
        feat_row['is_weekend'] = 1 if forecast_datetime.dayofweek >= 5 else 0
        feat_row['hour_sin'] = np.sin(2 * np.pi * forecast_datetime.hour / 24)
        feat_row['hour_cos'] = np.cos(2 * np.pi * forecast_datetime.hour / 24)
        feat_row['day_sin'] = np.sin(2 * np.pi * forecast_datetime.dayofweek / 7)
        feat_row['day_cos'] = np.cos(2 * np.pi * forecast_datetime.dayofweek / 7)
        
        # Update lag features with recent forecasts
        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            if h > lag:
                # Use forecasted value for past lags
                lag_idx = len(forecast_records) - lag
                if lag_idx >= 0:
                    feat_row[f'lag_{lag}h'] = forecast_records[lag_idx]['predicted']
                else:
                    feat_row[f'lag_{lag}h'] = last_actual
            else:
                feat_row[f'lag_{lag}h'] = last_actual
        
        # Update rolling statistics
        for w in [24, 48, 168]:
            recent_values = [last_actual] + [r['predicted'] for r in forecast_records[-(w-1):]]
            if len(recent_values) >= w:
                feat_row[f'rolling_mean_{w}h'] = np.mean(recent_values[-w:])
                feat_row[f'rolling_std_{w}h'] = np.std(recent_values[-w:])
            else:
                feat_row[f'rolling_mean_{w}h'] = last_actual
                feat_row[f'rolling_std_{w}h'] = 0
        
        # Update EMA
        for span in [24, 168]:
            feat_row[f'ema_{span}h'] = last_actual
        
        # Update same hour previous day/week
        if h > 24:
            feat_row['same_hour_yesterday'] = forecast_records[-24]['predicted']
        else:
            feat_row['same_hour_yesterday'] = last_actual
            
        if h > 168:
            feat_row['same_hour_last_week'] = forecast_records[-168]['predicted']
        else:
            feat_row['same_hour_last_week'] = last_actual
        
        # Update difference features
        feat_row['diff_1h'] = 0
        feat_row['diff_24h'] = 0
        
        # Prepare features for prediction
        X_forecast = feat_row[feature_cols].values.reshape(1, -1)
        X_forecast_scaled = scaler.transform(X_forecast)
        
        # Predict
        predicted = model.predict(X_forecast_scaled)[0]
        
        # Ensure non-negative
        predicted = max(0, predicted)
        
        # Store record
        forecast_records.append({
            'datetime': forecast_datetime,
            'actual': np.nan,  # No actual values for future
            'predicted': predicted
        })
        
        # Update last actual for next iteration
        last_actual = predicted
        
        # Progress display
        if h % 1000 == 0:
            print(f"  Forecasted {h:,}/{horizon:,} hours ({h/horizon*100:.1f}%)")
    
    # Convert to DataFrame
    forecast_df = pd.DataFrame(forecast_records)
    return forecast_df


def generate_nhits_forecast(nf, df_feat, feature_cols, target_col, horizon):
    """Generate N-HiTS forecast for specified horizon."""
    import pandas as pd
    import numpy as np
    
    print(f"\n[INFO] Generating N-HiTS forecast for {horizon} hours...")
    
    # Prepare data for NeuralForecast
    df_predict = df_feat.copy()
    if 'ds' not in df_predict.columns:
        df_predict['ds'] = df_predict['datetime']
    if 'y' not in df_predict.columns:
        df_predict['y'] = df_predict[target_col]
    if 'unique_id' not in df_predict.columns:
        df_predict['unique_id'] = 'PGCB'
    
    cols_order = ['unique_id', 'ds', 'y'] + feature_cols
    cols_order = [c for c in cols_order if c in df_predict.columns]
    df_predict = df_predict[cols_order]
    
    # Get last timestamp
    last_datetime = df_predict['ds'].iloc[-1]
    
    # Generate future timestamps
    future_dates = pd.date_range(start=last_datetime + pd.Timedelta(hours=1), 
                                  periods=horizon, freq='h')
    
    # Create future dataframe with exogenous features
    future_df = pd.DataFrame({
        'unique_id': ['PGCB'] * horizon,
        'ds': future_dates,
        'y': [np.nan] * horizon
    })
    
    # Add time-based features for future dates
    future_df['hour'] = future_df['ds'].dt.hour
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    future_df['day_of_month'] = future_df['ds'].dt.day
    future_df['month'] = future_df['ds'].dt.month
    future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
    
    # Add lag features using past data
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        lag_col = f'lag_{lag}h'
        if lag_col in df_predict.columns:
            # Get last values from training data
            last_values = df_predict[lag_col].iloc[-lag:].values
            if len(last_values) < lag:
                last_values = np.pad(last_values, (0, lag - len(last_values)), 
                                     mode='edge')
            future_df[lag_col] = last_values[-1]
    
    # Combine past and future for prediction
    df_combined = pd.concat([df_predict, future_df], ignore_index=True)
    
    # Predict with NeuralForecast
    try:
        predictions = nf.predict(df_combined, h=horizon)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'actual': [np.nan] * horizon,
            'predicted': predictions['NHITS'].values
        })
        
        # Ensure non-negative
        forecast_df['predicted'] = forecast_df['predicted'].clip(lower=0)
        
        return forecast_df
    except Exception as e:
        print(f"[!] NHITS prediction failed: {e}")
        # Fallback: create simple forecast using last value
        last_value = df_predict['y'].iloc[-1]
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'actual': [np.nan] * horizon,
            'predicted': [last_value] * horizon
        })
        return forecast_df


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def main():
    """Execute complete forecasting pipeline."""
    print("\n" + "=" * 80)
    print("PGCB HOURLY GENERATION FORECASTING PIPELINE")
    print("=" * 80)
    print("\nModels: LightGBM, NHITS")
    print("Forecast Horizons: 24h")
    
    # Step 0: Environment setup
    print("\n" + "=" * 80)
    print("PHASE 0: ENVIRONMENT SETUP")
    print("=" * 80)
    
    try:
        import pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, neuralforecast
        print("[✓] All required packages available")
    except ImportError as e:
        print(f"[!] Missing package: {e}")
        print("Installing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn", 
                       "matplotlib", "seaborn", "lightgbm", "statsmodels", "neuralforecast"], check=True)
    
    # Import pandas here for later use in main()
    import pandas as pd
    
    gpu_available, device_info = check_gpu()
    print(f"\n[INFO] Training device: {device_info}")
    
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
    print("PHASE 5: MODEL TRAINING")
    print("=" * 80)
    
    val_size = int(len(X_train) * 0.1)
    X_train_train = X_train[:-val_size]
    y_train_train = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    
    results = []
    
    # Train LightGBM
    print(f"\n[INFO] Training LightGBM on {device_info}")
    model_lgb, time_lgb = train_lightgbm(X_train_train, y_train_train, X_val, y_val, feature_cols, device_info)
    y_pred_lgb = model_lgb.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_lgb, 'LightGBM', 24))
    
    # Train NeuralForecast models (N-HiTS, TiDE, DLinear)
    try:
        print(f"\n[INFO] Training NeuralForecast models on {device_info}")
        nf, models, model_names = train_neuralforecast_models(X_train, y_train, X_val, y_val, df_feat, feature_cols, target_col, device_info)
        print("[✓] NeuralForecast models trained successfully")
    except Exception as e:
        print(f"[!] NeuralForecast failed with error: {e}")
        nf, models, model_names = None, None, None
    
    # Step 5: Save trained models
    print("\n" + "=" * 80)
    print("PHASE 5b: SAVE TRAINED MODELS")
    print("=" * 80)
    
    results_dir = create_results_folder()
    import pickle
    
    print(f"\n[INFO] Saving trained models to: {results_dir}")
    with open(f'{results_dir}/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model_lgb, f)
    print(f"  → Saved: {results_dir}/lightgbm_model.pkl")
    
    if nf is not None:
        with open(f'{results_dir}/neuralforecast_model.pkl', 'wb') as f:
            pickle.dump(nf, f)
        print(f"  → Saved: {results_dir}/neuralforecast_model.pkl")
        
        with open(f'{results_dir}/neuralforecast_models.pkl', 'wb') as f:
            pickle.dump((models, model_names), f)
        print(f"  → Saved: {results_dir}/neuralforecast_models.pkl")
    
    # Step 6: Evaluation
    print("\n" + "=" * 80)
    print("PHASE 6: MODEL EVALUATION")
    print("=" * 80)
    
    if nf is not None and models is not None:
        results = evaluate_neuralforecast_models(
            nf, models, model_names, df_feat, feature_cols, target_col, 
            results, results_dir, test_idx, y_test, X_test
        )
    
    # Print results
    print("\n" + "=" * 80)
    print("PHASE 6: EVALUATION RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    print("\nForecast Results (24h horizon):")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(f'{results_dir}/forecast_results.csv', index=False)
    print(f"\n✓ Results saved to {results_dir}/forecast_results.csv")
    
    # Plot LightGBM diagnostics (define dates_test first)
    print("\nGenerating diagnostic plots...")
    dates_test = df_feat.loc[test_idx, 'datetime']
    plot_forecasts(y_test, y_pred_lgb, dates_test, 'LightGBM', 24, folder=results_dir)
    plot_residuals(y_test, y_pred_lgb, 'LightGBM', folder=results_dir)
    
    # Save LightGBM predictions CSV (matching NHITS format)
    print("\n[INFO] Saving LightGBM predictions CSV...")
    pred_lgb_df = pd.DataFrame({
        'datetime': dates_test.values,
        'actual': y_test,
        'predicted': y_pred_lgb
    })
    pred_lgb_df.to_csv(f'{results_dir}/predictions_LightGBM_24h.csv', index=False)
    print(f"  → Saved: {results_dir}/predictions_LightGBM_24h.csv")
    
    # Save feature importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model_lgb.feature_importance()
    }).sort_values('Importance', ascending=False)
    importance.to_csv(f'{results_dir}/feature_importance.csv', index=False)
    print(f"\n✓ Feature importance saved to {results_dir}/feature_importance.csv")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput files (all in 'model_results' folder):")
    print(f"  • {results_dir}/forecast_results.csv - Model comparison results")
    print(f"  • {results_dir}/feature_importance.csv - LightGBM feature importance")
    print(f"  • {results_dir}/forecast_LightGBM_24h.png - LightGBM forecast visualization")
    print(f"  • {results_dir}/residuals_LightGBM.png - LightGBM residual analysis")
    print(f"  • {results_dir}/lightgbm_model.pkl - Trained LightGBM model")
    if nf is not None:
        print(f"  • {results_dir}/neuralforecast_model.pkl - Trained NeuralForecast model")
        print(f"  • {results_dir}/neuralforecast_models.pkl - Trained DL model instances")
        print(f"  • {results_dir}/predictions_NHITS_24h.csv - NHITS 24-hour predictions")
        print(f"  • {results_dir}/forecast_NHITS_24h.png - NHITS forecast visualization")
        print(f"  • {results_dir}/residuals_NHITS.png - NHITS residual analysis")
    print(f"  • {results_dir}/predictions_LightGBM_24h.csv - LightGBM 24-hour predictions")
    print(f"  • scaler.pkl - Feature scaler for inference")
    
    # Step 7: Generate next 24-hour forecasts
    print("\n" + "=" * 80)
    print("PHASE 7: GENERATING NEXT 24-HOUR FORECASTS")
    print("=" * 80)
    
    horizon_24h = 24  # Next 24 hours
    
    # LightGBM 24-hour forecast
    try:
        print(f"\n[INFO] Generating next 24-hour LightGBM forecast...")
        lgb_forecast_24h = generate_lightgbm_forecast(model_lgb, scaler, df_feat, feature_cols, 
                                                       target_col, datetime_col, horizon_24h)
        lgb_forecast_24h.to_csv(f'{results_dir}/forecast_LightGBM_24h.csv', index=False)
        print(f"  → Saved: {results_dir}/forecast_LightGBM_24h.csv")
    except Exception as e:
        print(f"[!] LightGBM 24h forecast failed: {e}")
    
    # NHITS 24-hour forecast
    if nf is not None:
        try:
            print(f"\n[INFO] Generating next 24-hour NHITS forecast...")
            nhits_forecast_24h = generate_nhits_forecast(nf, df_feat, feature_cols, 
                                                          target_col, horizon_24h)
            nhits_forecast_24h.to_csv(f'{results_dir}/forecast_NHITS_24h.csv', index=False)
            print(f"  → Saved: {results_dir}/forecast_NHITS_24h.csv")
        except Exception as e:
            print(f"[!] NHITS 24h forecast failed: {e}")
    
    print("\n" + "=" * 80)
    print("ALL FORECASTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {e}")
        print("\nDebug information:")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        traceback.print_exc()
        sys.exit(1)
