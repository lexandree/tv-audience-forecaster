import pandas as pd
import numpy as np
from typing import List, Tuple
from src.models.types import EventImpactProfile
from src.models.fft_reconstructor import reconstruct_from_fft

def generate_future_timestamps(start_date: str, days: int = 365) -> pd.DatetimeIndex:
    """
    Generates an hourly DatetimeIndex for the forecast period.
    """
    hours = days * 24
    return pd.date_range(start_date, periods=hours, freq='h')

def apply_event_impacts(
    forecast_df: pd.DataFrame, 
    future_events_df: pd.DataFrame, 
    impact_profiles: List[EventImpactProfile],
    demographic_key: Tuple[str, str]
) -> pd.DataFrame:
    """
    Applies the average historical residuals to the base forecast for scheduled future events.
    """
    df = forecast_df.copy()
    
    # 1. Filter profiles for this demographic
    demo_profiles = {p.event_category: p.average_residual_tvr for p in impact_profiles if p.demographic_key == demographic_key}
    
    if not demo_profiles or future_events_df.empty:
        return df
        
    # 2. Map future events to the forecast timeline
    events_indexed = future_events_df.set_index('timestamp')[['event_category']]
    events_indexed = events_indexed[~events_indexed.index.duplicated(keep='first')]
    
    merged = df.join(events_indexed, how='left')
    
    # 3. Apply the impacts
    for category, impact in demo_profiles.items():
        mask = merged['event_category'] == category
        merged.loc[mask, 'predicted_tvr'] += impact
        
    # Ensure no negative TVRs after adjustment
    merged['predicted_tvr'] = np.maximum(merged['predicted_tvr'], 0.0)
    
    # Clean up
    merged = merged.drop(columns=['event_category'])
    
    return merged
