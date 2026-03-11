import pytest
import pandas as pd
import numpy as np
from src.data.interpolation import seasonal_interpolate

def test_seasonal_interpolation():
    # Create an hourly timeseries of exactly 2 weeks (336 hours)
    dates = pd.date_range('2023-01-01', periods=336, freq='h')
    
    # Base values
    tvr = np.ones(336) * 5.0
    
    # Create missing values
    # Let's say hour 200 is missing. The same hour last week is 200 - 168 = 32.
    tvr[32] = 10.0 # Setting last week's value
    tvr[200] = np.nan # The missing value
    
    df = pd.DataFrame({'tvr': tvr}, index=dates)
    
    interpolated_df = seasonal_interpolate(df, 'tvr')
    
    # The missing value at index 200 should now be equal to the value at index 32
    assert not np.isnan(interpolated_df['tvr'].iloc[200])
    assert interpolated_df['tvr'].iloc[200] == 10.0
    
def test_seasonal_interpolation_consecutive_missing():
    # 3 weeks
    dates = pd.date_range('2023-01-01', periods=504, freq='h') 
    tvr = np.ones(504) * 5.0
    
    # Missing multiple weeks
    tvr[32] = 8.0 # Week 1 (value exists)
    tvr[200] = np.nan # Week 2 (missing)
    tvr[368] = np.nan # Week 3 (missing)
    
    df = pd.DataFrame({'tvr': tvr}, index=dates)
    interpolated_df = seasonal_interpolate(df, 'tvr')
    
    assert interpolated_df['tvr'].iloc[200] == 8.0
    assert interpolated_df['tvr'].iloc[368] == 8.0

