import pytest
import pandas as pd
from src.models.forecaster import generate_future_timestamps

def test_generate_future_timestamps():
    # 2026 is not a leap year, so 365 days = 8760 hours
    start_date = '2026-01-01 00:00:00'
    dates = generate_future_timestamps(start_date, days=365)
    
    assert len(dates) == 8760
    assert dates[0] == pd.to_datetime('2026-01-01 00:00:00')
    assert dates[-1] == pd.to_datetime('2026-12-31 23:00:00')
    
def test_generate_future_timestamps_leap_year():
    # 2024 is a leap year, so 366 days = 8784 hours
    start_date = '2024-01-01 00:00:00'
    dates = generate_future_timestamps(start_date, days=366)
    
    assert len(dates) == 8784
