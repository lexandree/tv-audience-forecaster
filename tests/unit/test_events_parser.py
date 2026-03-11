import pytest
import pandas as pd
from src.data.events_ingestion import map_events_to_timeseries

def test_map_events_to_timeseries():
    # 1. Mock timeseries index
    dates = pd.date_range('2023-12-24 18:00:00', periods=5, freq='h')
    df = pd.DataFrame({'tvr': [1, 2, 3, 4, 5]}, index=dates)
    
    # 2. Mock events calendar
    events_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-12-24 20:00:00', '2023-12-24 22:00:00']),
        'event_category': ['Christmas_Eve', 'Late_Movie'],
        'is_historical': [True, True]
    })
    
    # 3. Map events
    mapped_df = map_events_to_timeseries(df, events_df)
    
    assert 'event_category' in mapped_df.columns
    
    # Check that non-event times have None/NaN
    assert pd.isna(mapped_df.loc['2023-12-24 18:00:00', 'event_category'])
    
    # Check that event times are mapped correctly
    assert mapped_df.loc['2023-12-24 20:00:00', 'event_category'] == 'Christmas_Eve'
    assert mapped_df.loc['2023-12-24 22:00:00', 'event_category'] == 'Late_Movie'
