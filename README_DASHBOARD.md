# Power Demand Forecasting Dashboard

## Overview
This is an interactive Streamlit dashboard for visualizing power demand forecasting results from the PGCB dataset.

## Features
1. **Time-Series Visualization**: Interactive chart showing historical and forecasted power demand with confidence bands
2. **Summary Metrics**: Latest value, peak, average, and forecast horizon displayed in metric cards
3. **Model Error Metrics**: MAE, RMSE, and MAPE for LightGBM and NHITS models
4. **Interactive Controls**: Date range selector and model selector

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Install dependencies:
```bash
pip install streamlit pandas plotly
```

2. Run the dashboard:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## Usage

### Date Range Filter
- Use the date picker in the sidebar to select a date range
- The chart will update to show only data within the selected range

### Model Selector
- Select "All" to show both models
- Select "LightGBM" or "NHITS" to filter by specific model

## Data Files
The dashboard reads from the following files:
- `model_results/predictions_NHITS_24h.csv` - Hourly predictions
- `model_results/forecast_results.csv` - Model performance metrics
- `EDA_Output/summary_statistics.csv` - Dataset summary

## Project Structure
```
RA Assesment/
├── app.py                 # Streamlit dashboard application
├── requirements.txt       # Python dependencies
├── model_results/         # Forecast results and predictions
│   ├── predictions_NHITS_24h.csv
│   ├── forecast_results.csv
│   └── feature_importance.csv
├── EDA_Output/           # EDA output files
│   └── summary_statistics.csv
└── README_DASHBOARD.md   # This file
```

## Technologies Used
- **Streamlit**: Web framework for the dashboard
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualization library
- **Python**: Backend processing

## Notes
- The dashboard uses Streamlit's built-in caching for performance
- Chart includes 95% confidence bands around forecasted values
- Metric cards use gradient backgrounds for visual appeal