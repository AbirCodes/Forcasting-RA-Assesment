# Missing Values Handling Strategy

## Original Missing Values (From EDA)

| Feature | Missing Count | Missing % | Recommendation |
|---------|---------------|-----------|----------------|
| solar | 22,133 | 23.89% | Interpolate within days |
| wind | 73,974 | 79.84% | Drop (>50% missing) |
| india_adani | 85,312 | 92.08% | Drop (>50% missing) |
| nepal | 87,299 | 94.22% | Drop (>50% missing) |
| remarks | 86,257 | 93.10% | Drop (metadata) |

## What We Did

### 1. Target Variable: generation_mw
- **Missing values: 0** ✅
- No action needed - perfect for forecasting
- This is excellent data quality

### 2. Feature Selection

**Dropped (Missing > 50%):**
- `wind`: 79.84% missing
- `india_adani`: 92.08% missing
- `nepal`: 94.22% missing
- `remarks`: 93.10% missing

**Kept for Imputation:**
- `solar`: 23.89% missing - Below threshold, will interpolate
- `demand_mw`: Will check and impute if needed
- `load_shedding`: Will check and impute if needed
- `gas`, `liquid_fuel`, `coal`, `hydro`: Will check and impute if needed

### 3. Imputation Method

**For Numeric Columns:**
```python
# Step 1: Linear interpolation (preserves temporal patterns)
df[col] = df[col].interpolate(method='linear')

# Step 2: Forward fill (for initial values)
df[col] = df[col].fillna(method='ffill')

# Step 3: Backward fill (for final values)
df[col] = df[col].fillna(method='bfill')
```

**Why This Order:**
1. **Linear interpolation**: Best for temporal data, uses surrounding values
2. **FFill**: Handles missing values at the beginning of the series
3. **BFill**: Handles missing values at the end of the series

### 4. Implementation in Pipeline

```python
# In clean_data() function:
# 1. Drop columns with >50% missing
high_missing_cols = [col for col in columns if missing_pct[col] > 50]
df_clean = df_clean.drop(columns=high_missing_cols)

# 2. Impute remaining numeric columns
for col in numeric_columns:
    if df_clean[col].isnull().sum() > 0:
        # Linear interpolation
        df_clean[col] = df_clean[col].interpolate(method='linear')
        # Forward/backward fill for edges
        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
```

### 5. Example: Solar Power Imputation

If `solar` has 22,133 missing values (23.89%):
- Interpolation fills ~80-90% of gaps using adjacent hours
- FFill fills any remaining at start of dataset
- BFill fills any remaining at end of dataset
- **Result**: Complete solar column for modeling

### 6. Alternative Approaches Considered

| Method | Pros | Cons | Used |
|--------|------|------|------|
| Linear Interpolation | Temporal consistency, preserves patterns | Assumes linear trend between points | ✅ Primary |
| Mean/Median Imputation | Simple, fast | Loses temporal patterns | No |
| KNN Imputation | Considers similar observations | Computationally expensive | No |
| Model-based Imputation | Most accurate | Complex, requires extra model | No |
| Drop Rows | Simple | Reduces sample size significantly | No |

### 7. Impact on Forecasting

**Good News:**
- Target variable (generation_mw) has **no missing values**
- This is critical for supervised learning
- Temporal imputation preserves time-series structure

**Potential Issues:**
- Solar power has 23.89% missing - may affect solar-related features
- If solar is important for forecasting, consider:
  - Drop solar features entirely
  - Use separate solar forecasting model
  - Use weather data to predict solar generation

### 8. Verification

After imputation, the pipeline verifies:
```python
# Ensure no missing values in target
assert df_clean[target_col].isnull().sum() == 0

# Print imputation statistics
print(f"Features imputed: {imputed_count}/{len(numeric_cols)}")
```

### 9. Summary

**Missing Value Strategy:**
1. ✅ Drop columns with >50% missing (wind, imports, remarks)
2. ✅ Interpolate numeric columns with <50% missing (solar, etc.)
3. ✅ Use linear interpolation + FFill/BFill for temporal data
4. ✅ Verify target has no missing values before modeling

**Expected Outcome:**
- Clean dataset ready for feature engineering
- No missing values in features or target
- Temporal patterns preserved through interpolation
