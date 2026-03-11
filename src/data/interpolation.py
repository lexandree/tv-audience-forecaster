import pandas as pd
import numpy as np

def seasonal_interpolate(df: pd.DataFrame, column: str = 'tvr', seasonal_period_hours: int = 168) -> pd.DataFrame:
    """
    Fills missing values using the value from exactly one season ago.
    Default season is 1 week (168 hours).
    It will perform this iteratively to fill gaps larger than one week.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_interp = df.copy()
    
    # Fill forwards from previous weeks
    while df_interp[column].isna().any():
        shifted = df_interp[column].shift(seasonal_period_hours)
        df_interp[column] = df_interp[column].fillna(shifted)
        
        # If we can't fill anymore (e.g., missing values at the very beginning of the dataset),
        # we fallback to shifting backwards (from future weeks)
        if df_interp[column].isna().sum() == df[column].isna().sum():
            # If nothing changed, try shifting backwards
            shifted_back = df_interp[column].shift(-seasonal_period_hours)
            df_interp[column] = df_interp[column].fillna(shifted_back)
            
            # If still nothing changed, we are stuck (data too sparse)
            if df_interp[column].isna().sum() == df[column].isna().sum():
                # Ultimate fallback: linear interpolation for the remaining few
                df_interp[column] = df_interp[column].interpolate(method='linear')
                break
                
        df = df_interp.copy() # update baseline for next iteration
        
    return df_interp
