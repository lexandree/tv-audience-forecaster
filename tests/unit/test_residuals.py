import pytest
import pandas as pd
import numpy as np
from src.models.residuals import calculate_residuals

def test_calculate_residuals():
    dates = pd.date_range('2023-01-01', periods=4, freq='h')
    
    historical_df = pd.DataFrame({'tvr': [10.0, 15.0, 20.0, 25.0]}, index=dates)
    
    # Let's pretend our FFT model perfectly predicted the first two, 
    # but under-predicted the 3rd and over-predicted the 4th
    reconstructed = np.array([10.0, 15.0, 15.0, 30.0])
    
    residuals_df = calculate_residuals(historical_df, reconstructed)
    
    assert 'residual' in residuals_df.columns
    
    # Residual = Actual - Reconstructed
    assert residuals_df['residual'].iloc[0] == 0.0
    assert residuals_df['residual'].iloc[1] == 0.0
    assert residuals_df['residual'].iloc[2] == 5.0  # 20 - 15
    assert residuals_df['residual'].iloc[3] == -5.0 # 25 - 30
