import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.ingestion import load_audience_data, load_events_data
from src.utils.exceptions import ValidationError

def test_load_audience_data_valid(tmp_path):
    df = pd.DataFrame({
        'timestamp': ['2023-01-01 00:00:00', '2023-01-01 01:00:00'],
        'age_group': ['18-34', '18-34'],
        'gender': ['M', 'M'],
        'tvr': [2.5, 3.1]
    })
    filepath = tmp_path / "valid_audience.csv"
    df.to_csv(filepath, index=False)
    
    result = load_audience_data(filepath)
    assert len(result) == 2
    assert 'timestamp' in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

def test_load_audience_data_missing_columns(tmp_path):
    df = pd.DataFrame({
        'timestamp': ['2023-01-01 00:00:00'],
        'age_group': ['18-34']
        # missing gender and tvr
    })
    filepath = tmp_path / "invalid_audience.csv"
    df.to_csv(filepath, index=False)
    
    with pytest.raises(ValidationError):
        load_audience_data(filepath)

def test_load_audience_data_negative_tvr(tmp_path):
    df = pd.DataFrame({
        'timestamp': ['2023-01-01 00:00:00'],
        'age_group': ['18-34'],
        'gender': ['M'],
        'tvr': [-1.0]
    })
    filepath = tmp_path / "negative_tvr.csv"
    df.to_csv(filepath, index=False)
    
    with pytest.raises(ValidationError):
        load_audience_data(filepath)
