import pytest
import pandas as pd
from datetime import datetime
from src.data.segmentation import segment_by_demographic

def test_segment_by_demographic():
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00:00', '2023-01-01 00:00:00',
            '2023-01-01 01:00:00', '2023-01-01 01:00:00'
        ]),
        'age_group': ['18-24', '25-34', '18-24', '25-34'],
        'gender': ['M', 'M', 'M', 'M'],
        'tvr': [1.0, 2.0, 1.5, 2.5]
    })
    
    segments = segment_by_demographic(df)
    
    assert len(segments) == 2
    assert ('18-24', 'M') in segments
    assert ('25-34', 'M') in segments
    
    df_1824 = segments[('18-24', 'M')]
    assert len(df_1824) == 2
    assert df_1824.iloc[0]['tvr'] == 1.0
    
    # Ensure index is correctly set and sorted
    assert df_1824.index.name == 'timestamp'
    assert df_1824.index[0] == pd.to_datetime('2023-01-01 00:00:00')
