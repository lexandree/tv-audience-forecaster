import pandas as pd
from typing import List
from src.utils.exceptions import ValidationError

REQUIRED_AUDIENCE_COLUMNS = ['timestamp', 'age_group', 'gender', 'tvr']
REQUIRED_EVENTS_COLUMNS = ['timestamp', 'event_category', 'is_historical']

def validate_audience_data(df: pd.DataFrame) -> None:
    """Validates that the input data meets the required schema."""
    if df.empty:
        raise ValidationError("Input audience dataset is empty.")
        
    missing_cols = [col for col in REQUIRED_AUDIENCE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in audience data: {missing_cols}")
        
    # Attempt to convert timestamp
    try:
        pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValidationError(f"Invalid timestamp format in audience data: {e}")
        
    # Check for negative TVRs
    if (df['tvr'] < 0).any():
        raise ValidationError("Found negative TVR values. TVR must be >= 0.")

def validate_events_data(df: pd.DataFrame) -> None:
    """Validates that the events calendar meets the required schema."""
    if df.empty:
        # It's okay to have no events, but if a dataframe is passed and it's completely empty, we might want to warn.
        # But we won't raise an exception for empty events, as events are optional.
        return
        
    missing_cols = [col for col in REQUIRED_EVENTS_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in events data: {missing_cols}")
        
    try:
        pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValidationError(f"Invalid timestamp format in events data: {e}")
