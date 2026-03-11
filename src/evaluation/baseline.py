import pandas as pd
from src.utils.exceptions import ForecastingError

def generate_sply_baseline(historical_df: pd.DataFrame, forecast_dates: pd.DatetimeIndex, shift_days: int = 364) -> pd.DataFrame:
    """
    Generates a 'Same Period Last Year' baseline.
    By default uses a 364-day shift to ensure day-of-week alignment.
    """
    if 'tvr' not in historical_df.columns:
        raise ForecastingError("Historical dataframe must contain a 'tvr' column.")
        
    # We assume historical_df index is a DatetimeIndex
    sply_records = []
    
    for dt in forecast_dates:
        # Calculate exactly when "last year" was
        sply_dt = dt - pd.Timedelta(days=shift_days)
        
        # Look it up
        if sply_dt in historical_df.index:
            val = historical_df.loc[sply_dt, 'tvr']
            # Handle duplicates if the index isn't unique, though it should be
            if isinstance(val, pd.Series):
                val = val.iloc[0]
        else:
            val = float('nan') # Missing historical data
            
        sply_records.append({'timestamp': dt, 'tvr': val})
        
    sply_df = pd.DataFrame(sply_records).set_index('timestamp')
    return sply_df
