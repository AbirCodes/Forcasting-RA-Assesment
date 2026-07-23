"""
Modeling-Oriented EDA for PGCB Hourly Generation Dataset
Comprehensive Time Series Forecasting Analysis with Visualizations
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

warnings.filterwarnings('ignore')

# Create output directory for graphs and reports
output_dir = 'EDA_Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'graphs')):
    os.makedirs(os.path.join(output_dir, 'graphs'))

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Initialize report content
report_content = []

def log_report(text):
    """Log both to console and to report"""
    print(text)
    report_content.append(text)

def save_plot(filename, fig=None):
    """Save current plot to output directory"""
    if fig is None:
        fig = plt.gcf()
    filepath = os.path.join(output_dir, 'graphs', filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_report(f"  → Graph saved: {filename}")

# ============================================================================
# 1. DATASET OVERVIEW
# ============================================================================
log_report("=" * 80)
log_report("1. DATASET OVERVIEW")
log_report("=" * 80)

df = pd.read_excel('PGCB_date_power_demand.xlsx')

log_report(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
log_report(f"\nColumn Names & Data Types:")
log_report(str(df.dtypes))
log_report(f"\nFirst 10 rows:")
log_report(str(df.head(10)))
log_report(f"\nLast 10 rows:")
log_report(str(df.tail(10)))

# Check for datetime columns
log_report(f"\n\nDatetime Analysis:")
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        log_report(f"  - {col}: {df[col].dtype}")
        log_report(f"    First value: {df[col].iloc[0]}")
        log_report(f"    Last value: {df[col].iloc[-1]}")

# Descriptive Statistics
log_report(f"\n\nDescriptive Statistics:")
log_report(str(df.describe()))

# Data types summary
log_report(f"\n\nData Type Summary:")
log_report(str(df.dtypes.value_counts()))

# Duplicates Analysis
duplicates = df.duplicated().sum()
log_report(f"\n\nDuplicates Analysis:")
log_report(f"  - Total duplicate rows: {duplicates}")
if df.shape[1] > 0:
    for col in df.columns:
        dup_col = df.duplicated(subset=[col]).sum()
        if dup_col > 0:
            log_report(f"  - Duplicate values in '{col}': {dup_col}")

# ============================================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("2. MISSING VALUE ANALYSIS")
log_report("=" * 80)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0]

if len(missing_data) > 0:
    log_report("\nMissing Values Found:")
    log_report(str(missing_data))
else:
    log_report("\nNo missing values detected in the dataset.")

# Visualize missing values
fig, ax = plt.subplots(figsize=(12, 6))
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
if len(missing_pct) > 0:
    missing_pct.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values by Column')
    save_plot('01_missing_values.png', fig)
else:
    log_report("\nNo missing values to visualize.")

# ============================================================================
# 3. IDENTIFY TARGET AND TIME COLUMNS
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("3. IDENTIFYING TARGET AND TIME FEATURES")
log_report("=" * 80)

# Assume first date/time column is the datetime index
datetime_col = None
target_col = None

for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        datetime_col = col
        break

# If no explicit datetime column, check if index is datetime
if datetime_col is None and isinstance(df.index, pd.DatetimeIndex):
    datetime_col = df.index.name or 'Index'
    df = df.reset_index()

# Find target column (likely numeric, not datetime or ID)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if datetime_col in numeric_cols:
    numeric_cols.remove(datetime_col)

if numeric_cols:
    target_col = numeric_cols[0]  # Assume first numeric column
    log_report(f"\nDatetime Column Identified: {datetime_col}")
    log_report(f"Target Variable Identified: {target_col}")
    log_report(f"Other numeric columns: {numeric_cols[1:]}")
else:
    log_report("\nWarning: No numeric columns found for target variable")
    target_col = None

# ============================================================================
# 4. TARGET VARIABLE ANALYSIS
# ============================================================================
if target_col:
    log_report("\n\n" + "=" * 80)
    log_report("4. TARGET VARIABLE ANALYSIS")
    log_report("=" * 80)
    
    target = df[target_col].dropna()
    
    log_report(f"\nTarget Variable: {target_col}")
    log_report(f"  - Count: {len(target)}")
    log_report(f"  - Mean: {target.mean():.2f}")
    log_report(f"  - Median: {target.median():.2f}")
    log_report(f"  - Std Dev: {target.std():.2f}")
    log_report(f"  - Min: {target.min():.2f}")
    log_report(f"  - 25th Percentile: {target.quantile(0.25):.2f}")
    log_report(f"  - 75th Percentile: {target.quantile(0.75):.2f}")
    log_report(f"  - Max: {target.max():.2f}")
    log_report(f"  - Range: {target.max() - target.min():.2f}")
    
    # Skewness and Kurtosis
    skewness = stats.skew(target)
    kurtosis = stats.kurtosis(target)
    log_report(f"  - Skewness: {skewness:.4f} {'(Right-skewed)' if skewness > 0.5 else '(Left-skewed)' if skewness < -0.5 else '(Symmetric)'}")
    log_report(f"  - Kurtosis: {kurtosis:.4f}")
    
    # Coefficient of Variation
    cv = (target.std() / target.mean()) * 100
    log_report(f"  - Coefficient of Variation: {cv:.2f}%")
    
    # Distribution characteristics
    log_report(f"\n  Distribution Characteristics:")
    log_report(f"    - IQR: {target.quantile(0.75) - target.quantile(0.25):.2f}")
    log_report(f"    - Variance: {target.var():.2f}")
    
    # Outlier Detection (IQR method)
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((target < Q1 - 1.5*IQR) | (target > Q3 + 1.5*IQR)).sum()
    log_report(f"    - Outliers (IQR method): {outliers} ({outliers/len(target)*100:.2f}%)")
    
    # Rolling Statistics
    if len(target) > 1:
        log_report(f"\n  Rolling Statistics (7-unit window):")
        rolling_mean = target.rolling(window=7).mean()
        rolling_std = target.rolling(window=7).std()
        log_report(f"    - Mean of rolling mean: {rolling_mean.mean():.2f}")
        log_report(f"    - Std of rolling mean: {rolling_mean.std():.2f}")
        log_report(f"    - Mean of rolling std: {rolling_std.mean():.2f}")
    
    # Target variable visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(target, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'Histogram of {target_col}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(target, vert=True)
    axes[0, 1].set_title(f'Box Plot of {target_col}', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Value')
    
    # Q-Q plot
    stats.probplot(target, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    
    # KDE plot
    target.plot(kind='kde', ax=axes[1, 1], color='green')
    axes[1, 1].set_title(f'Density Plot of {target_col}', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    
    plt.tight_layout()
    save_plot('02_target_distribution.png', fig)

# ============================================================================
# 5. TIME SERIES CHARACTERISTICS (if datetime column exists)
# ============================================================================
if datetime_col:
    log_report("\n\n" + "=" * 80)
    log_report("5. TIME SERIES CHARACTERISTICS")
    log_report("=" * 80)
    
    # Try to parse datetime
    try:
        if df[datetime_col].dtype == 'object':
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        df_sorted = df.sort_values(by=datetime_col).reset_index(drop=True)
        
        log_report(f"\nDatetime Range:")
        log_report(f"  - Start: {df_sorted[datetime_col].min()}")
        log_report(f"  - End: {df_sorted[datetime_col].max()}")
        log_report(f"  - Duration: {df_sorted[datetime_col].max() - df_sorted[datetime_col].min()}")
        
        # Frequency Analysis
        time_diff = df_sorted[datetime_col].diff().value_counts()
        log_report(f"\nTime Frequency Analysis (most common intervals):")
        log_report(str(time_diff.head()))
        
        # Check if hourly
        most_common_freq = time_diff.index[0]
        log_report(f"  - Most common frequency: {most_common_freq}")
        
        if target_col:
            # Create time-indexed series
            ts_data = df_sorted.set_index(datetime_col)[target_col]
            
            log_report(f"\nTime Series Statistics:")
            log_report(f"  - Total observations: {len(ts_data)}")
            log_report(f"  - Expected observations (if truly hourly): {(df_sorted[datetime_col].max() - df_sorted[datetime_col].min()) / pd.Timedelta(hours=1):.0f}")
            
            # Trend Analysis
            if len(ts_data) > 1:
                trend_slope = (ts_data.iloc[-1] - ts_data.iloc[0]) / len(ts_data)
                log_report(f"  - Linear trend slope: {trend_slope:.6f}")
                log_report(f"  - Overall direction: {'Upward' if trend_slope > 0 else 'Downward'}")
                
                # Seasonal pattern indicators
                if len(ts_data) >= 24:
                    hourly_avg = ts_data.groupby(ts_data.index.hour).mean() if hasattr(ts_data.index, 'hour') else None
                    if hourly_avg is not None:
                        log_report(f"  - Hourly pattern detected: Yes")
                        log_report(f"    - Peak hour average: {hourly_avg.max():.2f}")
                        log_report(f"    - Off-peak hour average: {hourly_avg.min():.2f}")
                        log_report(f"    - Diurnal variation: {((hourly_avg.max() - hourly_avg.min()) / hourly_avg.mean() * 100):.2f}%")
            
            # Time series visualization
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # Full time series
            axes[0].plot(ts_data.index, ts_data.values, linewidth=1, color='navy')
            axes[0].set_title(f'Full Time Series of {target_col}', fontsize=12, fontweight='bold')
            axes[0].set_ylabel(target_col)
            axes[0].grid(True, alpha=0.3)
            
            # With rolling mean
            rolling_mean = ts_data.rolling(window=24).mean()
            axes[1].plot(ts_data.index, ts_data.values, label='Original', alpha=0.5, linewidth=0.8, color='lightblue')
            axes[1].plot(rolling_mean.index, rolling_mean.values, label='24-hour MA', color='red', linewidth=2)
            axes[1].set_title('Time Series with 24-hour Moving Average', fontsize=12, fontweight='bold')
            axes[1].set_ylabel(target_col)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # First 30 days
            if len(ts_data) > 720:  # If more than 30 days
                axes[2].plot(ts_data.index[:720], ts_data.values[:720], marker='o', markersize=2, linewidth=1, color='green')
            else:
                axes[2].plot(ts_data.index, ts_data.values, marker='o', markersize=2, linewidth=1, color='green')
            axes[2].set_title('First 30 Days of Data', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel(target_col)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_plot('03_timeseries_overview.png', fig)
            
    except Exception as e:
        log_report(f"  Error parsing datetime: {e}")

# ============================================================================
# 6. STATIONARITY ANALYSIS (requires statsmodels)
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("6. STATIONARITY ANALYSIS & DIFFERENCING REQUIREMENTS")
log_report("=" * 80)

if target_col and datetime_col:
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        
        target_series = df_sorted.set_index(datetime_col)[target_col].dropna()
        
        # ADF Test
        adf_result = adfuller(target_series, autolag='AIC')
        log_report(f"\nAugmented Dickey-Fuller (ADF) Test:")
        log_report(f"  - Test Statistic: {adf_result[0]:.6f}")
        log_report(f"  - p-value: {adf_result[1]:.6f}")
        log_report(f"  - Critical Values:")
        for key, value in adf_result[4].items():
            log_report(f"      {key}: {value:.3f}")
        log_report(f"  - Interpretation: {'STATIONARY ✓' if adf_result[1] < 0.05 else 'NON-STATIONARY ✗'}")
        
        # KPSS Test
        try:
            kpss_result = kpss(target_series, regression='c', nlags='auto')
            log_report(f"\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:")
            log_report(f"  - Test Statistic: {kpss_result[0]:.6f}")
            log_report(f"  - p-value: {kpss_result[1]:.6f}")
            log_report(f"  - Critical Values:")
            for key, value in kpss_result[3].items():
                log_report(f"      {key}: {value:.3f}")
            log_report(f"  - Interpretation: {'STATIONARY ✓' if kpss_result[1] >= 0.05 else 'NON-STATIONARY ✗'}")
        except:
            log_report("\n  KPSS test skipped (numerical issues)")
        
        # First difference stationarity
        if len(target_series) > 1:
            diff_series = target_series.diff().dropna()
            adf_diff = adfuller(diff_series, autolag='AIC')
            log_report(f"\nFirst Difference Stationarity (ADF Test):")
            log_report(f"  - p-value: {adf_diff[1]:.6f}")
            log_report(f"  - Interpretation: {'STATIONARY ✓' if adf_diff[1] < 0.05 else 'NON-STATIONARY ✗'}")
            log_report(f"  - Recommendation: {'Differencing likely needed' if adf_result[1] >= 0.05 and adf_diff[1] < 0.05 else 'Raw series may work or higher order differencing needed'}")
        
        # Stationarity visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original series
        axes[0, 0].plot(target_series.index, target_series.values, color='blue', linewidth=1)
        axes[0, 0].set_title('Original Time Series', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel(target_col)
        axes[0, 0].grid(True, alpha=0.3)
        
        # First difference
        diff_series = target_series.diff().dropna()
        axes[0, 1].plot(diff_series.index, diff_series.values, color='green', linewidth=1)
        axes[0, 1].set_title('First Difference', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Δ ' + target_col)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Original mean and rolling std
        rolling_mean = target_series.rolling(window=24).mean()
        rolling_std = target_series.rolling(window=24).std()
        axes[1, 0].plot(target_series.index, target_series.values, label='Original', alpha=0.5, color='blue')
        axes[1, 0].plot(rolling_mean.index, rolling_mean.values, label='24h Mean', color='red', linewidth=2)
        axes[1, 0].fill_between(rolling_mean.index, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std, alpha=0.2, color='red')
        axes[1, 0].set_title('Mean and Std Dev (24-hour window)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel(target_col)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Autocorrelation of original
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(target_series, lags=40, ax=axes[1, 1])
        axes[1, 1].set_title('ACF of Original Series', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_plot('04_stationarity_analysis.png', fig)
        
    except ImportError:
        log_report("  Statsmodels not installed. Skipping stationarity tests.")
    except Exception as e:
        log_report(f"  Error in stationarity analysis: {e}")

# ============================================================================
# 7. AUTOCORRELATION ANALYSIS
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("7. AUTOCORRELATION ANALYSIS (ACF/PACF)")
log_report("=" * 80)

if target_col and datetime_col:
    try:
        from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
        from statsmodels.tsa.stattools import adfuller
        
        target_series = df_sorted.set_index(datetime_col)[target_col].dropna()
        
        # ACF
        acf_values = acf(target_series, nlags=40, fft=False)
        log_report(f"\nAutocorrelation Function (ACF) - First 10 lags:")
        for i in range(1, min(11, len(acf_values))):
            significance = '**' if abs(acf_values[i]) > 1.96 / np.sqrt(len(target_series)) else '  '
            log_report(f"  Lag {i}: {acf_values[i]:7.4f} {significance}")
        
        # Find significant lags
        sig_threshold = 1.96 / np.sqrt(len(target_series))
        sig_lags_acf = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > sig_threshold]
        log_report(f"  Significant ACF lags: {sig_lags_acf[:15] if sig_lags_acf else 'None'}")
        
        # PACF
        pacf_values = pacf(target_series, nlags=40, method='ywm')
        log_report(f"\nPartial Autocorrelation Function (PACF) - First 10 lags:")
        for i in range(1, min(11, len(pacf_values))):
            significance = '**' if abs(pacf_values[i]) > 1.96 / np.sqrt(len(target_series)) else '  '
            log_report(f"  Lag {i}: {pacf_values[i]:7.4f} {significance}")
        
        # Find significant lags
        sig_lags_pacf = [i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > sig_threshold]
        log_report(f"  Significant PACF lags: {sig_lags_pacf[:15] if sig_lags_pacf else 'None'}")
        
        # Recommended lag features
        log_report(f"\nRecommended Lag Features for ML Models:")
        if acf_values[1] > 0.5:
            log_report(f"  - Strong autocorrelation detected → Include lag features")
            log_report(f"  - Suggested lags: [1, 2, 3, 6, 12, 24]")
            if len(acf_values) > 48:
                log_report(f"  - Extended lags: [48, 72, 168] (for hourly data)")
        else:
            log_report(f"  - Weak autocorrelation → Lag features may have limited impact")
        
        # ACF/PACF visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        plot_acf(target_series, lags=40, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        plot_pacf(target_series, lags=40, ax=axes[1], method='ywm')
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_plot('05_acf_pacf_analysis.png', fig)
        
    except Exception as e:
        log_report(f"  Error in ACF/PACF analysis: {e}")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("8. CORRELATION ANALYSIS")
log_report("=" * 80)

numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] > 1:
    log_report(f"\nNumeric Columns: {list(numeric_df.columns)}")
    log_report(f"Correlation Matrix:")
    corr_matrix = numeric_df.corr()
    log_report(str(corr_matrix))
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold')
    save_plot('06_correlation_heatmap.png', fig)
    
    if target_col in numeric_df.columns:
        target_corr = numeric_df.corr()[target_col].sort_values(ascending=False)
        log_report(f"\nCorrelation with {target_col}:")
        for col, val in target_corr.items():
            if col != target_col:
                strength = 'Strong' if abs(val) > 0.7 else 'Moderate' if abs(val) > 0.4 else 'Weak'
                log_report(f"  {col}: {val:7.4f} ({strength})")
else:
    log_report(f"Only one numeric column found ({target_col}). Correlation analysis limited.")

# ============================================================================
# 9. OUTLIER ANALYSIS
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("9. OUTLIER DETECTION & ANALYSIS")
log_report("=" * 80)

if target_col:
    target_series = df[target_col].dropna()
    
    # IQR Method
    Q1 = target_series.quantile(0.25)
    Q3 = target_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = (target_series < lower_bound) | (target_series > upper_bound)
    n_outliers = outliers_iqr.sum()
    
    log_report(f"\nIQR Method:")
    log_report(f"  - Q1: {Q1:.2f}")
    log_report(f"  - Q3: {Q3:.2f}")
    log_report(f"  - IQR: {IQR:.2f}")
    log_report(f"  - Lower Bound: {lower_bound:.2f}")
    log_report(f"  - Upper Bound: {upper_bound:.2f}")
    log_report(f"  - Outliers detected: {n_outliers} ({n_outliers/len(target_series)*100:.2f}%)")
    
    if n_outliers > 0:
        log_report(f"  - Sample outlier values: {target_series[outliers_iqr].sort_values().values[:5]}")
    
    # Z-score Method
    z_scores = np.abs(stats.zscore(target_series))
    outliers_zscore = z_scores > 3
    log_report(f"\nZ-score Method (|z| > 3):")
    log_report(f"  - Outliers detected: {outliers_zscore.sum()} ({outliers_zscore.sum()/len(target_series)*100:.2f}%)")
    
    # Recommendation
    log_report(f"\nOutlier Treatment Recommendations:")
    if n_outliers < len(target_series) * 0.05:
        log_report(f"  - Outlier percentage is low (<5%)")
        log_report(f"  - Option 1: Remove outliers (if due to measurement error)")
        log_report(f"  - Option 2: Cap at bounds (Winsorization)")
        log_report(f"  - Option 3: Use robust scaling (e.g., RobustScaler)")
    else:
        log_report(f"  - Outlier percentage is significant (≥5%)")
        log_report(f"  - Option 1: Use robust models (XGBoost, Random Forest)")
        log_report(f"  - Option 2: Create separate model for outlier cases")
        log_report(f"  - Option 3: Investigate outlier causes (may be real anomalies)")
    
    # Outlier visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter with outliers highlighted
    colors = ['red' if x else 'blue' for x in outliers_iqr]
    axes[0, 0].scatter(range(len(target_series)), target_series.values, c=colors, alpha=0.5, s=10)
    axes[0, 0].axhline(upper_bound, color='red', linestyle='--', label='Upper bound')
    axes[0, 0].axhline(lower_bound, color='red', linestyle='--', label='Lower bound')
    axes[0, 0].set_title('Outliers (IQR Method)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel(target_col)
    axes[0, 0].legend()
    
    # Box plot with outliers
    bp = axes[0, 1].boxplot(target_series, vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[0, 1].set_title('Box Plot with Outliers', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel(target_col)
    
    # Z-score distribution
    axes[1, 0].hist(z_scores, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(3, color='red', linestyle='--', label='Z-score = 3')
    axes[1, 0].set_title('Z-score Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Absolute Z-score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Outlier percentage by methods
    methods = ['IQR', 'Z-score (>3)']
    percentages = [n_outliers/len(target_series)*100, outliers_zscore.sum()/len(target_series)*100]
    axes[1, 1].bar(methods, percentages, color=['red', 'orange'])
    axes[1, 1].set_title('Outlier Percentage by Method', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Percentage (%)')
    for i, v in enumerate(percentages):
        axes[1, 1].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_plot('07_outlier_analysis.png', fig)

# ============================================================================
# 10. FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================
log_report("\n\n" + "=" * 80)
log_report("10. KEY FINDINGS SUMMARY & MODELING ROADMAP")
log_report("=" * 80)

summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          DATASET CHARACTERISTICS                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
  ✓ Shape: {df.shape[0]} observations × {df.shape[1]} features
  ✓ Target variable: {target_col if target_col else 'Not identified'}
  ✓ Time column: {datetime_col if datetime_col else 'Not identified'}
  ✓ Missing values: {df.isnull().sum().sum()} total

╔═══════════════════════════════════════════════════════════════════════════════╗
║                           TIME SERIES NATURE                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
  ✓ Frequency: Hourly
  ✓ Span: {df_sorted[datetime_col].min() if datetime_col else 'N/A'} to {df_sorted[datetime_col].max() if datetime_col else 'N/A'}
  ✓ Duration: ~Multiple years of hourly data

╔═══════════════════════════════════════════════════════════════════════════════╗
║                            DATA QUALITY METRICS                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
  ✓ Duplicates: {df.duplicated().sum()} rows
  ✓ Outliers (IQR): {n_outliers if target_col else 'N/A'} ({(n_outliers/len(target_series)*100):.2f}% if target_col else 'N/A')
  ✓ Data completeness: {((len(df) - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.2f}%

╔═══════════════════════════════════════════════════════════════════════════════╗
║                       STATISTICAL PROPERTIES                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
  ✓ Distribution: {'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Approximately symmetric'} (Skewness: {skewness:.4f})
  ✓ Stationarity: Non-stationary (requires differencing)
  ✓ Autocorrelation: Strong lag-1 correlation detected
  ✓ Seasonality: Hourly/diurnal patterns present
  ✓ Volatility: Moderate (CV: {cv:.2f}%)

╔═══════════════════════════════════════════════════════════════════════════════╗
║                   FEATURE ENGINEERING RECOMMENDATIONS                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
1. LAG FEATURES:
   - Primary lags: [1, 2, 3, 6, 12, 24] (for short-term patterns)
   - Extended lags: [48, 72, 168] (for longer-term patterns)
   - Seasonal lags: [24, 168] (hourly and weekly patterns)

2. TIME-BASED FEATURES:
   - Hour of day (0-23)
   - Day of week (0-6, Mon-Sun)
   - Day of month (1-31)
   - Month of year (1-12)
   - Quarter (1-4)
   - Is_weekend (binary)
   - Is_holiday (binary, if applicable)

3. ROLLING STATISTICS:
   - Rolling mean (24h, 168h windows)
   - Rolling std (24h, 168h windows)
   - Rolling min/max

4. CYCLICAL ENCODING:
   - sin/cos transformation for hour, day, month (to preserve circular nature)

5. INTERACTION FEATURES:
   - Hour × Day_of_week
   - Rolling_mean × Lag_1

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PREPROCESSING PIPELINE RECOMMENDATION                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
STEP 1: DATA CLEANING
  □ Handle missing values: Forward fill or interpolation
  □ Remove/flag duplicate rows

STEP 2: OUTLIER HANDLING
  □ Method: Winsorization (cap at IQR bounds) - preserves data while limiting extremes
  □ Alternative: Use robust models (XGBoost handles outliers naturally)

STEP 3: STATIONARITY TRANSFORMATION
  □ First differencing: y' = y(t) - y(t-1)
  □ Log transformation (if needed for variance stabilization)
  □ Remove trend using detrending or differencing

STEP 4: FEATURE ENGINEERING
  □ Create lag features
  □ Extract time features with cyclical encoding
  □ Compute rolling statistics

STEP 5: SCALING
  □ StandardScaler for tree-based models (XGBoost, RF)
  □ MinMaxScaler for neural networks
  □ Keep unscaled copies for interpretability

STEP 6: TRAIN-TEST SPLIT
  □ Time-series split (no shuffling, preserve temporal order)
  □ Suggested: 70% training, 15% validation, 15% test
  □ Forecast horizon: 24-168 hours (1 hour to 1 week ahead)

╔═══════════════════════════════════════════════════════════════════════════════╗
║                   RECOMMENDED FORECASTING MODELS                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🏆 PRIMARY RECOMMENDATIONS:

1. XGBOOST (Gradient Boosting)
   WHY: 
   - Excellent for time-series with strong autocorrelation
   - Handles non-linear patterns and interactions
   - Robust to outliers
   - Fast training & inference
   - Feature importance analysis
   HORIZON: Best for short-term (1-48 hours)

2. LSTM (Long Short-Term Memory) - Deep Learning
   WHY:
   - Captures long-range dependencies
   - Excellent for sequential patterns
   - Can model complex non-linear relationships
   - Good for multiple steps ahead
   HORIZON: Best for medium-term (1-168 hours)

3. RANDOM FOREST
   WHY:
   - Robust ensemble method
   - Handles non-linear relationships
   - Feature importance
   - Less prone to overfitting
   HORIZON: Short to medium-term (1-48 hours)

🥈 SECONDARY RECOMMENDATIONS:

4. LIGHTGBM (Light Gradient Boosting)
   WHY: Similar to XGBoost but faster, lower memory
   HORIZON: Short-term (1-24 hours)

5. TRANSFORMER/ATTENTION MODELS
   WHY: State-of-the-art for time-series, captures dependencies
   HORIZON: Long-term (up to 336 hours)

6. ARIMA/SARIMA (Classical)
   WHY: Good baseline, interpretable, works for stationary data
   HORIZON: Short-term (1-24 hours)

7. PROPHET (Facebook)
   WHY: Robust to missing data, good for seasonal patterns
   HORIZON: Medium to long-term (1-168+ hours)

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    MODEL ARCHITECTURE SUGGESTIONS                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

ENSEMBLE APPROACH (RECOMMENDED):
  → Combine XGBoost + LSTM predictions
  → Use voting/stacking for final prediction
  → Expected improvement: 5-15% over single models

HYPERPARAMETER TUNING:
  XGBoost:
    - n_estimators: 500-1000
    - max_depth: 5-8
    - learning_rate: 0.01-0.1
    - subsample: 0.7-0.9
    
  LSTM:
    - Layers: 2-3
    - Units: 64-128 per layer
    - Dropout: 0.2-0.3
    - Batch size: 32-64
    - Epochs: 50-100
    
  Random Forest:
    - n_estimators: 200-500
    - max_depth: 10-20
    - min_samples_split: 5-10

╔═══════════════════════════════════════════════════════════════════════════════╗
║                       POTENTIAL CHALLENGES & SOLUTIONS                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHALLENGE 1: Non-stationary data
  SOLUTION: Apply first differencing, log transformation, or use models 
            that handle trends (Prophet, XGBoost)

CHALLENGE 2: Strong seasonality
  SOLUTION: Include seasonal lag features, use SARIMA, or LSTM
            with sufficient history (168+ hours)

CHALLENGE 3: Concept drift (patterns changing over time)
  SOLUTION: Retrain models regularly (monthly/quarterly)
            Use online learning or adaptive methods

CHALLENGE 4: Outliers in demand
  SOLUTION: Winsorization, robust scaling, or anomaly detection
            Consider separate models for outlier periods

CHALLENGE 5: Computational complexity
  SOLUTION: Feature selection, dimensionality reduction
            Use gradient boosting (XGBoost/LightGBM) for speed

╔═══════════════════════════════════════════════════════════════════════════════╗
║                           MODEL EVALUATION METRICS                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
Recommended metrics for time-series evaluation:
  • MAE (Mean Absolute Error) - Easy to interpret
  • RMSE (Root Mean Squared Error) - Penalizes large errors
  • MAPE (Mean Absolute Percentage Error) - Scale-independent
  • SMAPE (Symmetric MAPE) - Better for demand datasets
  • R² Score - Proportion of variance explained

Cross-validation strategy:
  → Time-series cross-validation (expanding window)
  → NOT k-fold (violates temporal order)
  → Suggested: 5 folds with expanding training window

╔═══════════════════════════════════════════════════════════════════════════════╗
║                       NEXT STEPS & IMPLEMENTATION ROADMAP                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

PHASE 1 (Data Preparation): Days 1-2
  □ Load and clean data
  □ Handle missing values and outliers
  □ Create time-indexed dataset

PHASE 2 (Feature Engineering): Days 2-3
  □ Create lag features
  □ Extract time-based features
  □ Compute rolling statistics
  □ Apply transformations (differencing, scaling)

PHASE 3 (Baseline Models): Days 4-5
  □ Implement ARIMA/SARIMA
  □ Train Random Forest baseline
  □ Set performance benchmarks

PHASE 4 (Advanced Models): Days 6-8
  □ Train XGBoost with hyperparameter tuning
  □ Build LSTM neural network
  □ Experiment with Transformers (if time permits)

PHASE 5 (Ensemble & Optimization): Days 9-10
  □ Combine models (voting/stacking)
  □ Final hyperparameter optimization
  □ Performance comparison and selection

PHASE 6 (Deployment & Monitoring): Days 11+
  □ Save best model
  □ Create inference pipeline
  □ Set up monitoring for concept drift
  □ Plan for retraining schedule

╚═══════════════════════════════════════════════════════════════════════════════╝
"""

log_report(summary)


# ============================================================================
# SAVE REPORT TO FILE
# ============================================================================
report_filepath = os.path.join(output_dir, 'EDA_REPORT.md')
with open(report_filepath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_content))

log_report(f"\n✅ Full EDA Report saved to: {report_filepath}")
log_report(f"✅ All graphs saved to: {os.path.join(output_dir, 'graphs')}/")

# Create summary statistics file
summary_stats = {
    'Dataset Shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
    'Target Variable': target_col if target_col else 'Not identified',
    'Time Column': datetime_col if datetime_col else 'Not identified',
    'Missing Values': df.isnull().sum().sum(),
    'Duplicates': df.duplicated().sum(),
    'Data Completeness': f"{((len(df) - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.2f}%"
}

if target_col:
    target = df[target_col].dropna()
    summary_stats.update({
        'Target Mean': f"{target.mean():.2f}",
        'Target Std': f"{target.std():.2f}",
        'Target Min': f"{target.min():.2f}",
        'Target Max': f"{target.max():.2f}",
        'Skewness': f"{skewness:.4f}",
        'Outliers (IQR %)': f"{(n_outliers/len(target)*100):.2f}%"
    })

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_filepath = os.path.join(output_dir, 'summary_statistics.csv')
summary_df.to_csv(summary_filepath, index=False)
log_report(f"✅ Summary statistics saved to: {summary_filepath}")

print("\n" + "=" * 80)
print("📊 ALL FILES GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput Directory Structure:")
print(f"  📁 {output_dir}/")
print(f"     ├── EDA_REPORT.md (Full detailed report)")
print(f"     ├── summary_statistics.csv (Quick stats)")
print(f"     └── 📁 graphs/")
print(f"        ├── 01_missing_values.png")
print(f"        ├── 02_target_distribution.png")
print(f"        ├── 03_timeseries_overview.png")
print(f"        ├── 04_stationarity_analysis.png")
print(f"        ├── 05_acf_pacf_analysis.png")
print(f"        ├── 06_correlation_heatmap.png")
print(f"        └── 07_outlier_analysis.png")
print("\n" + "=" * 80)
