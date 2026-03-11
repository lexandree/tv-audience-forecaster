import pytest
import pandas as pd
from src.data.export import export_forecast
import os

def test_export_forecast_contract(tmp_path):
    # Create sample forecast
    dates = pd.date_range('2026-01-01', periods=2, freq='h')
    df1 = pd.DataFrame({
        'timestamp': dates,
        'age_group': '18-34',
        'gender': 'M',
        'predicted_tvr': [2.5, 3.1]
    })
    df2 = pd.DataFrame({
        'timestamp': dates,
        'age_group': '18-34',
        'gender': 'F',
        'predicted_tvr': [1.5, 4.1]
    })
    
    final_df = pd.concat([df1, df2], ignore_index=True)
    
    out_path = tmp_path / "forecast.csv"
    export_forecast(final_df, str(out_path))
    
    assert os.path.exists(out_path)
    
    read_back = pd.read_csv(out_path)
    assert len(read_back) == 4
    assert list(read_back.columns) == ['timestamp', 'age_group', 'gender', 'predicted_tvr']
    assert read_back['age_group'].iloc[0] == '18-34'
