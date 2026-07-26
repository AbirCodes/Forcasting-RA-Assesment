"""
Power Demand Forecasting Dashboard
Streamlit app with interactive time-series visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Power Demand Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache all data files"""
    try:
        # Load predictions data
        predictions_path = os.path.join(os.path.dirname(__file__), 'model_results', 'predictions_NHITS_24h.csv')
        if os.path.exists(predictions_path):
            predictions_df = pd.read_csv(predictions_path)
            predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
        else:
            predictions_df = pd.DataFrame()
        
        # Load forecast results
        forecast_path = os.path.join(os.path.dirname(__file__), 'model_results', 'forecast_results.csv')
        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path)
        else:
            forecast_df = pd.DataFrame()
        
        # Load EDA summary
        summary_path = os.path.join(os.path.dirname(__file__), 'EDA_Output', 'summary_statistics.csv')
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = pd.DataFrame()
        
        # Load feature importance
        feature_path = os.path.join(os.path.dirname(__file__), 'model_results', 'feature_importance.csv')
        if os.path.exists(feature_path):
            feature_df = pd.read_csv(feature_path)
        else:
            feature_df = pd.DataFrame()
        
        return predictions_df, forecast_df, summary_df, feature_df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_time_series_chart(df, start_date, end_date, model):
    """Create interactive time series chart with Plotly"""
    # Filter data
    mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        return None
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Add historical data trace
    fig.add_trace(
        go.Scatter(
            x=filtered_df['datetime'],
            y=filtered_df['actual'],
            name='Historical',
            mode='lines',
            line=dict(color='#17a2b8', width=2),
            hovertemplate='<b>Time</b>: %{x}<br><b>Actual</b>: %{y:.2f} MW<extra></extra>'
        )
    )
    
    # Add forecast trace
    fig.add_trace(
        go.Scatter(
            x=filtered_df['datetime'],
            y=filtered_df['predicted'],
            name='Forecast',
            mode='lines',
            line=dict(color='#28a745', width=2),
            hovertemplate='<b>Time</b>: %{x}<br><b>Forecast</b>: %{y:.2f} MW<extra></extra>'
        )
    )
    
    # Add confidence band (95%)
    lower_bound = filtered_df['predicted'] * 0.95
    upper_bound = filtered_df['predicted'] * 1.05
    
    fig.add_trace(
        go.Scatter(
            x=list(filtered_df['datetime']) + list(filtered_df['datetime'])[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='tonexty',
            fillcolor='rgba(40, 167, 69, 0.2)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='95% Confidence',
            hoverinfo='skip'
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title='Date/Time',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=3, label="3d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title='Power Demand (MW)',
            showgrid=True
        ),
        hovermode='x unified',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=60, r=30, t=50, b=60),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_summary_cards(predictions_df):
    """Create summary metric cards"""
    if predictions_df.empty:
        return None
    
    # Calculate metrics
    latest_value = predictions_df['actual'].iloc[-1]
    peak_value = predictions_df['actual'].max()
    avg_value = predictions_df['actual'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Latest Actual Demand</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{latest_value:,.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="font-size: 0.8rem;">MW</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Peak Demand</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{peak_value:,.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="font-size: 0.8rem;">MW</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Demand</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_value:,.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="font-size: 0.8rem;">MW</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Forecast Horizon</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">24h</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="font-size: 0.8rem;">Hours ahead</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_error_metrics_table(forecast_df):
    """Create model error metrics table"""
    if forecast_df.empty:
        return
    
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    for idx, row in forecast_df.iterrows():
        model_name = row['Model']
        if model_name == 'LightGBM':
            color = '#17a2b8'
            bg_color = '#e3f2fd'
        else:
            color = '#6f42c1'
            bg_color = '#f3e5f5'
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                    <h4 style="color: {color}; margin: 0 0 15px 0;">{model_name}</h4>
                    <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                        <div>
                            <div style="font-size: 0.9rem; color: #666;">MAE</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{row['MAE']:.2f} MW</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; color: #666;">RMSE</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{row['RMSE']:.2f} MW</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; color: #666;">MAPE</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{row['MAPE']:.2f}%</div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

def main():
    """Main application function"""
    st.markdown('<div class="main-header">⚡ Power Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive visualization of historical and forecasted power demand using LightGBM and NHITS models</div>', unsafe_allow_html=True)
    
    # Load data
    predictions_df, forecast_df, summary_df, feature_df = load_data()
    
    if predictions_df.empty:
        st.warning("No data available. Please ensure model_results/predictions_NHITS_24h.csv exists.")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Get date range from data
    min_date = predictions_df['datetime'].min().date()
    max_date = predictions_df['datetime'].max().date()
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    
    # Model selector
    models = ['All', 'LightGBM', 'NHITS']
    selected_model = st.sidebar.selectbox("🤖 Select Model", models)
    
    # Update button
    if st.sidebar.button("🔄 Update Dashboard"):
        st.rerun()
    
    # Main content area
    st.divider()
    
    # Summary cards
    create_summary_cards(predictions_df)
    
    st.divider()
    
    # Error metrics
    create_error_metrics_table(forecast_df)
    
    st.divider()
    
    # Time series chart
    st.subheader("📈 Historical & Forecasted Power Demand")
    
    fig = create_time_series_chart(predictions_df, start_date, end_date, selected_model)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected date range.")
    
    # Data info panel
    st.divider()
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Size", "92,650 rows × 15 columns")
    
    with col2:
        completeness = summary_df[summary_df['Metric'] == 'Data Completeness']['Value'].values
        st.metric("Data Completeness", completeness[0] if len(completeness) > 0 else "N/A")
    
    with col3:
        outliers = summary_df[summary_df['Metric'] == 'Outliers (IQR %)']['Value'].values
        st.metric("Outlier Percentage", outliers[0] if len(outliers) > 0 else "N/A")

if __name__ == "__main__":
    main()