import pandas as pd
from typing import Dict, Tuple

from src.data.ingestion import load_audience_data
from src.data.segmentation import segment_by_demographic
from src.data.interpolation import seasonal_interpolate

def process_historical_data(filepath: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Complete pipeline for historical data:
    1. Ingestion & Validation
    2. Segmentation by Age/Gender
    3. Seasonal Interpolation (Gap filling)
    """
    # 1. Ingest
    df = load_audience_data(filepath)
    
    # 2. Segment
    segments = segment_by_demographic(df)
    
    # 3. Interpolate
    processed_segments = {}
    for key, segment_df in segments.items():
        processed_df = seasonal_interpolate(segment_df)
        processed_segments[key] = processed_df
        
    return processed_segments
