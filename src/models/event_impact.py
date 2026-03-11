import pandas as pd
from typing import List, Tuple
from src.models.types import EventImpactProfile

def build_impact_profile(df_with_residuals: pd.DataFrame, demographic_key: Tuple[str, str]) -> List[EventImpactProfile]:
    """
    Groups the dataframe by event_category and calculates the mean residual impact.
    Returns a list of EventImpactProfiles.
    """
    profiles = []
    
    # Filter only rows that actually have an event mapped
    events_only = df_with_residuals.dropna(subset=['event_category'])
    
    if events_only.empty:
        return profiles
        
    grouped = events_only.groupby('event_category')['residual'].mean()
    
    for category, mean_res in grouped.items():
        prof = EventImpactProfile(
            demographic_key=demographic_key,
            event_category=category,
            average_residual_tvr=float(mean_res)
        )
        profiles.append(prof)
        
    return profiles
