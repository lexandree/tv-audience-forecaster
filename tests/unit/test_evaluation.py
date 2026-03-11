import pytest
import pandas as pd
import numpy as np
from src.evaluation.baseline import generate_sply_baseline
from src.evaluation.metrics import calculate_mape

def test_calculate_mape():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([9.0, 22.0, 30.0])
    
    # Errors: |10-9|/10 = 0.1, |20-22|/20 = 0.1, |30-30|/30 = 0.0
    # Mean error = 0.2 / 3 = 0.0666...
    mape = calculate_mape(actual, predicted)
    assert np.isclose(mape, 0.066666, atol=0.001)

def test_generate_sply_baseline():
    # 2 years of data
    dates = pd.date_range('2023-01-01', periods=17520, freq='h')
    df = pd.DataFrame({'tvr': np.arange(17520)}, index=dates)
    
    # We want to forecast for 2024-01-01 to 2024-01-07
    forecast_dates = pd.date_range('2024-01-01', periods=168, freq='h')
    
    sply = generate_sply_baseline(df, forecast_dates, shift_days=364)
    
    assert len(sply) == 168
    
    # The SPLY for 2024-01-01 00:00:00 should be the value exactly 364 days prior:
    # 2024-01-01 is index 8760. 364 days = 8736 hours. 8760 - 8736 = 24.
    # So the value should be whatever was at index 24.
    expected_val = df.iloc[24]['tvr']
    assert sply.iloc[0]['tvr'] == expected_val
