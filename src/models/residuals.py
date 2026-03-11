import pandas as pd
import numpy as np

def calculate_residuals(actual_df: pd.DataFrame, reconstructed_signal: np.ndarray) -> pd.DataFrame:
    """
    Calculates the difference between the actual observed TVR and the
    FFT reconstructed baseline.
    
    Positive residual = Event increased viewership.
    Negative residual = Event decreased viewership.
    """
    df = actual_df.copy()
    df['reconstructed_tvr'] = reconstructed_signal
    df['residual'] = df['tvr'] - df['reconstructed_tvr']
    return df
