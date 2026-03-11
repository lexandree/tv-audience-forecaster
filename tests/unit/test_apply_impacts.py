import pytest
import pandas as pd
import numpy as np
from src.models.types import EventImpactProfile
from src.models.forecaster import apply_event_impacts

def test_apply_event_impacts():
    # 1. Base forecast
    dates = pd.date_range('2026-01-01', periods=5, freq='h')
    base_forecast_df = pd.DataFrame({
        'predicted_tvr': [10.0, 10.0, 10.0, 10.0, 10.0],
        'age_group': '18-34',
        'gender': 'M'
    }, index=dates)
    
    # 2. Future events calendar
    events_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 01:00:00', '2026-01-01 03:00:00']),
        'event_category': ['Football', 'News'],
        'is_historical': [False, False]
    })
    
    # 3. Impact Profiles
    profiles = [
        EventImpactProfile(demographic_key=('18-34', 'M'), event_category='Football', average_residual_tvr=5.0),
        EventImpactProfile(demographic_key=('18-34', 'M'), event_category='News', average_residual_tvr=-2.0),
        EventImpactProfile(demographic_key=('55+', 'F'), event_category='Football', average_residual_tvr=0.0) # Should be ignored
    ]
    
    # Apply
    adjusted_df = apply_event_impacts(base_forecast_df, events_df, profiles, ('18-34', 'M'))
    
    assert adjusted_df['predicted_tvr'].iloc[0] == 10.0 # No event
    assert adjusted_df['predicted_tvr'].iloc[1] == 15.0 # Football (+5)
    assert adjusted_df['predicted_tvr'].iloc[2] == 10.0 # No event
    assert adjusted_df['predicted_tvr'].iloc[3] == 8.0  # News (-2)
    assert adjusted_df['predicted_tvr'].iloc[4] == 10.0 # No event
