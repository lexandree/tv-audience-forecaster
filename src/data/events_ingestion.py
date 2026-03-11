import pandas as pd

def map_events_to_timeseries(timeseries_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins the events calendar onto the hourly timeseries data.
    Ensures that every hour in the timeseries either has an event_category or NaN.
    """
    df = timeseries_df.copy()
    
    # The events_df must have a 'timestamp' column
    # We will set it as index to do a join
    events_indexed = events_df.set_index('timestamp')[['event_category']]
    
    # Handle possible multiple events at the exact same hour by picking the first one
    # or joining them. For simplicity, we drop duplicates.
    events_indexed = events_indexed[~events_indexed.index.duplicated(keep='first')]
    
    # Join onto the main timeseries index
    merged_df = df.join(events_indexed, how='left')
    
    return merged_df
