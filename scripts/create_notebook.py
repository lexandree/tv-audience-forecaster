import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown Cell 1
md_intro = """# TV Audience Forecast Analysis
Interactive exploratory data analysis (EDA) and validation report for the FFT forecasting model.

This notebook:
1. Re-runs the core model to extract historical errors (residuals).
2. Computes an empirical confidence band based on historical variance by `(Day of Week, Hour)`.
3. Plots an interactive chart with Plotly, allowing you to zoom through time and select specific target audiences.
"""

# Code Cell 1: Setup and Imports
code_imports = """import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.display import display

# Import our custom pipeline
import sys
import os
sys.path.append('../') # Ensure src/ is accessible

from src.data.pipeline import process_historical_data
from src.models.pipeline import train_fft_models
from src.models.fft_reconstructor import reconstruct_from_fft
from src.models.residuals import calculate_residuals

# Suppress pandas warnings for clean output
import warnings
warnings.filterwarnings('ignore')"""

# Code Cell 2: Load Data and Compute Errors
code_data = """# 1. Load Historical Data & Train Model to get Residuals (Errors)
print("Loading history and training model to compute validation errors...")
hist_segments = process_historical_data('../data/raw/synthetic_history.csv')
fft_profiles = train_fft_models(hist_segments, k=100)

# 2. Load the 2026 Forecast
print("Loading 2026 Forecast...")
forecast_df = pd.read_csv('../data/output/forecast_2026.csv')
forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

# 3. Compute empirical error bands
confidence_bands = {}
for key, df in hist_segments.items():
    # Get historical reconstructed baseline
    profile = fft_profiles[key]
    reconstructed = reconstruct_from_fft(profile, length=len(df))
    
    # Calculate residuals (Actual - Predicted)
    res_df = calculate_residuals(df, reconstructed)
    
    # Group errors by Day of Week and Hour to find typical variance
    res_df['dayofweek'] = res_df.index.dayofweek
    res_df['hour'] = res_df.index.hour
    
    # Calculate standard deviation of error for each specific time slot
    error_std = res_df.groupby(['dayofweek', 'hour'])['residual'].std().reset_index()
    error_std.rename(columns={'residual': 'std_error'}, inplace=True)
    
    confidence_bands[key] = error_std

print("Data ready for plotting!")"""

# Code Cell 3: Plotly Visualization
code_plot = """# 4. Build Interactive Plotly Figure
fig = go.Figure()

target_audiences = forecast_df[['age_group', 'gender']].drop_duplicates().values.tolist()
target_audiences = [tuple(x) for x in target_audiences]

# We will create traces for ALL demographics, but only show the first one initially.
buttons = []

for i, (age, gender) in enumerate(target_audiences):
    is_visible = (i == 0) # Only the first audience is visible by default
    key = (age, gender)
    
    # Filter forecast
    f_df = forecast_df[(forecast_df['age_group'] == age) & (forecast_df['gender'] == gender)].copy()
    
    # Merge the historical error bands onto the future forecast dates
    f_df['dayofweek'] = f_df['timestamp'].dt.dayofweek
    f_df['hour'] = f_df['timestamp'].dt.hour
    f_df = pd.merge(f_df, confidence_bands[key], on=['dayofweek', 'hour'], how='left')
    
    # Upper and Lower bounds (e.g., +/- 1 Standard Deviation)
    f_df['upper_bound'] = f_df['predicted_tvr'] + f_df['std_error']
    f_df['lower_bound'] = np.maximum(f_df['predicted_tvr'] - f_df['std_error'], 0) # No negative TVR
    
    # TRACE 1: Upper Bound (invisible line to shape the fill)
    fig.add_trace(go.Scatter(
        x=f_df['timestamp'], y=f_df['upper_bound'],
        mode='lines', line=dict(width=0),
        showlegend=False, visible=is_visible,
        name=f"{age} {gender} Upper", hoverinfo='skip'
    ))
    
    # TRACE 2: Lower Bound (filled area back to Trace 1)
    fig.add_trace(go.Scatter(
        x=f_df['timestamp'], y=f_df['lower_bound'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', # Light blue transparent
        showlegend=True, visible=is_visible,
        name=f"Historical Error Band (+/- 1 StdDev)"
    ))
    
    # TRACE 3: Main Forecast Line
    fig.add_trace(go.Scatter(
        x=f_df['timestamp'], y=f_df['predicted_tvr'],
        mode='lines', line=dict(color='rgb(0, 176, 246)', width=2),
        showlegend=True, visible=is_visible,
        name=f"Forecasted TVR"
    ))
    
    # Create the dropdown button logic
    # Each audience has 3 traces. We create a boolean array of what should be visible.
    visibility = [False] * (len(target_audiences) * 3)
    visibility[i*3 : i*3+3] = [True, True, True]
    
    button = dict(
        label=f"{age} {gender}",
        method="update",
        args=[{"visible": visibility},
              {"title": f"TV Audience Forecast 2026 - Target: {age} {gender}"}]
    )
    buttons.append(button)

# Configure layout
fig.update_layout(
    title=f"TV Audience Forecast 2026 - Target: {target_audiences[0][0]} {target_audiences[0][1]}",
    xaxis_title="Date",
    yaxis_title="TVR (%)",
    updatemenus=[
        dict(
            active=0,
            buttons=buttons,
            x=1.15,
            y=1.05,
            xanchor="right",
            yanchor="top"
        )
    ],
    hovermode="x unified", # Shows values for line and band at the same X coordinate
    template="plotly_white",
    xaxis=dict(
        rangeslider=dict(visible=True), # Adds a small zoom slider at the bottom
        type="date"
    )
)

fig.show()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_code_cell(code_plot)
]

with open('notebooks/01_Forecast_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully at notebooks/01_Forecast_Analysis.ipynb")
