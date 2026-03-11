import pytest
import pandas as pd
from src.models.event_impact import build_impact_profile

def test_build_impact_profile():
    # Mock data with residuals and mapped events
    dates = pd.date_range('2023-01-01', periods=5, freq='h')
    df = pd.DataFrame({
        'tvr': [10, 10, 10, 10, 10],
        'residual': [0.0, 5.0, 0.0, 7.0, -1.0],
        'event_category': [None, 'Football', None, 'Football', 'News']
    }, index=dates)
    
    demographic_key = ('18-34', 'M')
    
    profiles = build_impact_profile(df, demographic_key)
    
    # We should have profiles for 'Football' and 'News'
    assert len(profiles) == 2
    
    # Football average residual: (5.0 + 7.0) / 2 = 6.0
    football_prof = next(p for p in profiles if p.event_category == 'Football')
    assert football_prof.average_residual_tvr == 6.0
    assert football_prof.demographic_key == demographic_key
    
    # News average residual: -1.0 / 1 = -1.0
    news_prof = next(p for p in profiles if p.event_category == 'News')
    assert news_prof.average_residual_tvr == -1.0
