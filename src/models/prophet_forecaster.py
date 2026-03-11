import pandas as pd
import numpy as np
from prophet import Prophet
import logging

def train_prophet_model(df: pd.DataFrame, regressors: pd.DataFrame = None) -> Prophet:
    """
    Trains a Prophet model on the provided historical demographic DataFrame.
    The DataFrame is expected to have 'timestamp' index and 'tvr' column,
    or at least 'tvr' and the index represents timestamps.
    
    If regressors are provided (e.g., holidays, weather), they must have the same index 
    and columns matching the regressor names.
    """
    # Prepare data for Prophet which requires 'ds' and 'y' columns
    prophet_df = pd.DataFrame()
    if 'timestamp' in df.columns:
        prophet_df['ds'] = df['timestamp']
    else:
        prophet_df['ds'] = df.index
        
    prophet_df['y'] = df['tvr'].values
    
    # Initialize Prophet with daily and weekly seasonality
    # Yearly seasonality is important too but might require >1 year of data. 
    # Auto-detection is usually fine, but we can be explicit.
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality='auto'
    )
    
    if regressors is not None:
        # Add regressors to the model and the dataframe
        for col in regressors.columns:
            if col != 'timestamp' and col != 'ds':
                model.add_regressor(col)
                prophet_df[col] = regressors[col].values

    # Silence Prophet's chatty output if possible
    import logging as py_logging
    py_logging.getLogger('prophet').setLevel(py_logging.WARNING)

    model.fit(prophet_df)
    return model

def predict_prophet_model(model: Prophet, future_dates: pd.DatetimeIndex, future_regressors: pd.DataFrame = None) -> np.ndarray:
    """
    Predicts future TVR using a trained Prophet model.
    """
    future_df = pd.DataFrame({'ds': future_dates})
    
    if future_regressors is not None:
        for col in future_regressors.columns:
            if col != 'timestamp' and col != 'ds':
                # Map by index or assume aligned order. Assuming aligned order for simplicity 
                # if future_dates and future_regressors are aligned.
                # Better to merge on 'ds'/timestamp:
                reg_copy = future_regressors.copy()
                if 'timestamp' in reg_copy.columns:
                    reg_copy = reg_copy.rename(columns={'timestamp': 'ds'})
                elif reg_copy.index.name == 'timestamp' or isinstance(reg_copy.index, pd.DatetimeIndex):
                    reg_copy['ds'] = reg_copy.index
                    
                future_df = future_df.merge(reg_copy[['ds', col]], on='ds', how='left')
                # Fill missing regressors with 0 or a sensible default
                future_df[col] = future_df[col].fillna(0)

    forecast = model.predict(future_df)
    
    # Ensure no negative predictions
    predicted_tvr = np.maximum(forecast['yhat'].values, 0.0)
    
    return predicted_tvr
