# PGCB Hourly Generation Forecasting Pipeline

Complete implementation with GPU support, multi-horizon forecasting, and state-of-the-art models.

## 🚀 Quick Start

### Windows
```cmd
setup_and_run.bat
```

### Linux/Mac
```bash
python forecast_pipeline_complete.py
```

This will:
1. ✅ Automatically install all dependencies
2. ✅ Detect GPU availability
3. ✅ Clean and preprocess the data
4. ✅ Engineer features
5. ✅ Train models (LightGBM, N-BEATS, TCN, TFT, Informer)
6. ✅ Evaluate and save results

---

## 📋 Requirements

### Hardware
- **CPU**: Minimum (but slow for deep learning)
- **GPU**: NVIDIA with CUDA support (recommended for fast training)
  - PyTorch: CUDA 11.8+
  - TensorFlow: CUDA 11.8+

### Software
- Python 3.8+
- pip package manager

### Key Python Packages
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn, statsmodels
- lightgbm, xgboost (gradient boosting)
- torch, tensorflow (deep learning)
- neuralforecast (N-BEATS, TCN, TFT, Informer)
- optuna (hyperparameter tuning)

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 0: SETUP                            │
│  • Environment setup (virtual env if needed)                │
│  • GPU detection and configuration                          │
│  • Package installation (auto if missing)                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1: DATA LOADING                      │
│  • Load PGCB dataset                                        │
│  • Identify target variable (generation_mw)                 │
│  • Explore data structure                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2: DATA CLEANING                    │
│  • Remove duplicate rows                                    │
│  • Convert timestamps to datetime index                     │
│  • Handle missing values (linear interpolation)             │
│  • Handle outliers (cap at domain knowledge + IQR)          │
│  • Verify hourly frequency                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 3: FEATURE ENGINEERING                │
│  Time Features:                                               │
│    • Hour, day_of_week, month, quarter                      │
│    • is_weekend, is_holiday                                 │
│    • Cyclical encoding (sin/cos)                            │
│  Lag Features:                                                │
│    • [1, 2, 3, 6, 12, 24, 48, 72, 168] hours                │
│    • Same hour yesterday/last week                          │
│  Rolling Statistics:                                          │
│    • Mean, std, min, max (24h, 48h, 168h windows)           │
│  EMA Features:                                                │
│    • Exponential moving averages (24h, 168h)                │
│  Difference Features:                                         │
│    • diff_1h, diff_24h                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 4: DATA SPLITTING                    │
│  • Chronological split (80% train / 20% test)               │
│  • NO SHUFFLING (preserve temporal order)                   │
│  • Feature scaling (StandardScaler)                         │
│  • Save scaler for inference                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 5: MODEL TRAINING                    │
│                                                               │
│  LightGBM:                                                    │
│    • Gradient boosted trees                                 │
│    • Fast training, excellent for tabular data              │
│    • Handles non-linear relationships                       │
│                                                               │
│  NeuralForecast Models:                                       │
│    • N-BEATS: Neural Basis Expansion Analysis               │
│    • TCN: Temporal Convolutional Network                    │
│    • TFT: Temporal Fusion Transformer                       │
│    • Informer: Efficient Transformer for long series        │
│                                                               │
│  Baseline:                                                    │
│    • ARIMA/SARIMA for comparison                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 6: EVALUATION                         │
│  Metrics:                                                     │
│    • MAE (Mean Absolute Error)                              │
│    • RMSE (Root Mean Squared Error)                         │
│    • MAPE (Mean Absolute Percentage Error)                  │
│                                                               │
│  Visualizations:                                              │
│    • Forecast vs Actual plots                               │
│    • Residual analysis (histogram, Q-Q plot)                │
│    • Feature importance                                     │
└─────────────────────��───────────────────────────────────────┘
```

---

## 📊 Target Variables & Forecast Horizons

### Target Variable
- **generation_mw**: Total hourly power generation in MW

### Forecast Horizons
- **1-hour ahead**: Short-term forecasting
- **6-hour ahead**: Medium-term forecasting
- **24-hour ahead**: Daily forecasting

### Multi-step Forecasting Strategy
- **Direct Multi-step**: Train separate model for each horizon (recommended)
- **Recursive**: Use predictions as inputs for next steps
- **Encoder-Decoder**: N-BEATS, TFT, Informer support this natively

---

## 🔍 Data Cleaning Details

### 1. Duplicate Removal
```python
# Drop completely duplicate rows
df = df.drop_duplicates()
```

### 2. Missing Value Imputation
```python
# Linear interpolation for gaps
df[target_col] = df[target_col].interpolate(method='linear')
# Forward/backward fill for edge cases
df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')
```

### 3. Outlier Handling
```python
# Domain knowledge: Bangladesh capacity ~24,000 MW
# Statistical: IQR-based bounds
upper_bound = min(IQR_upper, 30000)
df.loc[df[target_col] > upper_bound, target_col] = upper_bound
```

### 4. Frequency Verification
```python
# Ensure consistent hourly frequency
df = df.resample('H').mean()
df = df.interpolate()
```

---

## 🛠️ Feature Engineering

### Time Features
| Feature | Description | Purpose |
|---------|-------------|---------|
| hour | 0-23 | Captures daily pattern |
| day_of_week | 0-6 (Mon-Sun) | Captures weekly pattern |
| month | 1-12 | Captures seasonal pattern |
| quarter | 1-4 | Captures quarterly patterns |
| is_weekend | Binary | Weekend effect |
| day_of_year | 1-366 | Annual pattern |

### Cyclical Encoding
```python
# Preserves circular nature (hour 23→0 is close, not far)
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```

### Lag Features
| Lag | Purpose |
|-----|---------|
| 1h | Short-term dependency |
| 24h | Daily pattern |
| 48h | Two-day pattern |
| 168h | Weekly pattern |

### Rolling Statistics
| Window | Features |
|--------|----------|
| 24h | Mean, std, min, max |
| 48h | Mean, std, min, max |
| 168h | Mean, std, min, max |

### EMA Features
| Span | Weight | Purpose |
|------|--------|---------|
| 24h | Recent | Smooth intra-day noise |
| 168h | Older | Capture weekly trend |

---

## 🎯 Model Specifications

### LightGBM
```python
Parameters:
  • boosting_type: 'gbdt'
  • num_leaves: 31
  • learning_rate: 0.05
  • feature_fraction: 0.8
  • bagging_fraction: 0.8
  • bagging_freq: 5
  • min_child_samples: 20
  • max_iterations: 500
  • early_stopping: 50 rounds
```

### N-BEATS (Neural Basis Expansion Analysis)
- Architecture: Stack of fully-connected layers
- Decomposition: Trend + Seasonality
- Output: Multi-step forecast (24 steps)
- Training: ~100 epochs

### TCN (Temporal Convolutional Network)
- Architecture: Dilated causal convolutions
- Key features: Long-range dependencies
- Output: Multi-step forecast
- Training: ~100 epochs

### TFT (Temporal Fusion Transformer)
- Architecture: LSTM + Multi-head attention
- Key features: Interpretability + performance
- Output: Multi-step forecast with attention weights
- Training: ~100 epochs

### Informer
- Architecture: Efficient Transformer (prob Sparse attention)
- Key features: Long-range forecasting
- Output: Multi-step forecast
- Training: ~100 epochs

---

## 📈 Evaluation Metrics

### MAE (Mean Absolute Error)
```python
MAE = mean(|y_true - y_pred|)
```
- Interpretation: Average absolute error
- Units: MW
- Lower is better

### RMSE (Root Mean Squared Error)
```python
RMSE = sqrt(mean((y_true - y_pred)^2))
```
- Interpretation: Standard deviation of errors
- Units: MW
- Penalizes large errors more than MAE
- Lower is better

### MAPE (Mean Absolute Percentage Error)
```python
MAPE = mean(|(y_true - y_pred) / y_true|) * 100
```
- Interpretation: Average percentage error
- Units: %
- Scale-independent
- Handle division by zero carefully

---

## 📁 Output Files

### Results
- `forecast_results.csv` - Model comparison table
- `feature_importance.csv` - LightGBM feature importance

### Visualizations
- `forecast_LightGBM_24h.png` - Forecast plot
- `forecast_ARIMA_24h.png` - Baseline forecast plot
- `residuals_LightGBM.png` - Residual analysis
- `residuals_ARIMA.png` - Baseline residuals

### Artifacts
- `scaler.pkl` - Feature scaler (for inference)
- `model_LightGBM.pkl` - Trained LightGBM model (if saved)

---

## 🚀 GPU Setup (Recommended)

### PyTorch with CUDA
```bash
# Windows/Linux
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### TensorFlow with GPU
```bash
# Install with CUDA support
pip install tensorflow[and-cuda]

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### CUDA Toolkit
Download from: https://developer.nvidia.com/cuda-downloads

---

## 📊 Expected Performance

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Training Time |
|-------|----------|-----------|----------|---------------|
| LightGBM | ~1,000-1,500 | ~1,400-2,000 | ~10-15% | ~1-2 min |
| ARIMA | ~1,500-2,500 | ~2,000-3,000 | ~15-20% | ~5-10 min |
| N-BEATS | ~900-1,300 | ~1,200-1,800 | ~9-12% | ~5-10 min |
| TCN | ~850-1,200 | ~1,100-1,600 | ~8-11% | ~5-10 min |
| TFT | ~800-1,100 | ~1,000-1,500 | ~7-10% | ~10-15 min |
| Informer | ~800-1,100 | ~1,000-1,500 | ~7-10% | ~10-15 min |

**Expected improvement over baseline: 20-40%**

---

## 🔧 Troubleshooting

### Issue: "No module named 'xxx'"
**Solution**: Packages not installed
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or model size
```python
# In neural network training
batch_size = 16  # Reduce from 32/64
```

### Issue: "Data contains NaN values"
**Solution**: Data not cleaned properly
```python
df = df.dropna()
# Or use imputation
df = df.fillna(method='ffill')
```

### Issue: "Model training too slow"
**Solution**: 
- Use GPU (see GPU setup above)
- Reduce dataset size (temporarily)
- Reduce model complexity

---

## 📝 Customization

### Change Forecast Horizon
```python
# In create_features()
create_features(df, target_col, datetime_col, forecast_horizon=24)

# In NeuralForecast
h = 6  # 6-hour horizon
```

### Change Train/Test Split
```python
# In create_train_test_split()
create_train_test_split(df, feature_cols, target_col, test_size=0.2)
# Change 0.2 to 0.15 (85/15 split)
```

### Add New Features
```python
# In create_features()
# Add your custom features
df_feat['custom_feature'] = ...
```

---

## 🎓 Academic References

- **N-BEATS**: Oreshkin et al. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- **TCN**: Bai et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **TFT**: Kim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **Informer**: Zhou et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

---

## 📄 License

This project is for educational and research purposes.

---

**Author**: Data Science Team  
**Date**: June 2025  
**Dataset**: PGCB Hourly Generation (Bangladesh Power System)  
**Version**: 1.0
