import pandas as pd
from typing import Optional
from src.data.validators import validate_audience_data, validate_events_data

def load_audience_data(filepath: str) -> pd.DataFrame:
    """
    Loads historical TV audience data from a CSV file.
    Validates the schema before returning.
    """
    df = pd.read_csv(filepath)
    validate_audience_data(df)
    
    # Ensure timestamp is datetime and tvr is float32
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['tvr'] = df['tvr'].astype('float32')
    
    return df

def load_events_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads external events calendar.
    """
    try:
        df = pd.read_csv(filepath)
        validate_events_data(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return None
