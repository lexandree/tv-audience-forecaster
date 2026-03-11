import pandas as pd
from typing import Dict, Tuple

def segment_by_demographic(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Groups a full audience dataframe by age and gender into separate dataframes.
    Sets the timestamp as the index and sorts it.
    
    Returns:
        Dict mapping (age_group, gender) -> DataFrame(index=timestamp, columns=[tvr])
    """
    segments = {}
    grouped = df.groupby(['age_group', 'gender'])
    
    for (age, gender), group_df in grouped:
        # Sort by timestamp
        group_df = group_df.sort_values('timestamp')
        
        # Set timestamp as index
        group_df = group_df.set_index('timestamp')
        
        # Keep only the target metric
        segments[(age, gender)] = group_df[['tvr']]
        
    return segments
