import pandas as pd
import requests

def fetch_dwd_icon_weather(start_date: str, end_date: str, lat: float = 51.1657, lon: float = 10.4515) -> pd.DataFrame:
    """
    Fetches DWD ICON forecast/historical data from Open-Meteo for a specific coordinate (default central Germany).
    Used for creating short-term weather adjustment multipliers.
    """
    # Use Open-Meteo historical/forecast API (simplified to archive for history)
    # Note: To support both history and future seamlessly in a real prod app, 
    # we would route to 'archive-api' vs 'api' based on the date.
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation"],
        "timezone": "Europe/Berlin"
    }
    
    response = requests.get(url, params=params)
    
    # If the dates are in the future, the archive API might fail, so we fallback to the forecast API.
    if response.status_key != 200 or 'error' in response.json():
        url_forecast = "https://api.open-meteo.com/v1/forecast"
        response = requests.get(url_forecast, params=params)
        
    response.raise_for_status()
    data = response.json()
    
    hourly = data['hourly']
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        'temperature_2m': hourly['temperature_2m'],
        'precipitation': hourly['precipitation']
    }).set_index('timestamp')
    
    return df

def apply_weather_adjustments(df: pd.DataFrame, weather_df: pd.DataFrame, base_column: str = 'tvr') -> pd.DataFrame:
    """
    Applies heuristics to adjust TVR based on weather.
    - Extreme Heat (> 28C) or Heavy Rain (> 10mm) -> +9% TVR (multiplier 1.09)
    - Extreme Cold (< -5C) -> +6% TVR (multiplier 1.06)
    - Perfect Outdoor Weather (20C - 25C, 0 rain) -> -5% TVR (multiplier 0.95)
    """
    merged = df.join(weather_df, how='left')
    
    multiplier = pd.Series(1.0, index=merged.index)
    
    # Apply conditions
    multiplier.loc[(merged['temperature_2m'] > 28) | (merged['precipitation'] > 10)] = 1.09
    multiplier.loc[merged['temperature_2m'] < -5] = 1.06
    multiplier.loc[(merged['temperature_2m'] >= 20) & (merged['temperature_2m'] <= 25) & (merged['precipitation'] == 0)] = 0.95
    
    df_adjusted = df.copy()
    df_adjusted[f'{base_column}_adjusted'] = merged[base_column] * multiplier
    
    return df_adjusted
