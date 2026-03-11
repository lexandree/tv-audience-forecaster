import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse

# Real AGF Videoforschung base minutes (as percentage of viewing)
REAL_BASE_MINUTES = {
    '18-24 M': 40, 
    '18-24 F': 45, 
    '25-34 M': 80, 
    '25-34 F': 85,
    '55+ M': 280, 
    '55+ F': 320
}

def generate_demographic_data(demographic, start_date, end_date):
    """
    Generates hourly synthetic TVR data for a specific demographic, 
    incorporating daily and weekly seasonality.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='h')[:-1] # hourly
    n_hours = len(dates)
    
    # Base TVR from minutes (rough heuristic: base_minutes / 3.0)
    base_tvr = REAL_BASE_MINUTES.get(demographic, 50) / 3.0
    
    # Time components
    hours = dates.hour.values
    dayofweek = dates.dayofweek.values
    
    # 1. Daily Seasonality (Prime time peak around 20:00 - 22:00)
    daily_wave = np.cos((hours - 21) * (2 * np.pi / 24)) 
    # Scale it
    daily_effect = daily_wave * (base_tvr * 0.4) 
    
    # 2. Weekly Seasonality (Higher on Sunday (6), lower on weekdays)
    weekly_wave = np.cos((dayofweek - 6) * (2 * np.pi / 7))
    weekly_effect = weekly_wave * (base_tvr * 0.15)
    
    # 3. Noise
    noise = np.random.normal(0, base_tvr * 0.1, n_hours)
    
    # Combine
    tvr = base_tvr + daily_effect + weekly_effect + noise
    
    # Ensure no negative ratings
    tvr = np.maximum(tvr, 0)
    
    # Add some missing values (simulate dropouts)
    missing_mask = np.random.rand(n_hours) < 0.01 # 1% missing
    tvr[missing_mask] = np.nan
    
    # Split demographic key
    age_group, gender = demographic.split(' ')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'age_group': age_group,
        'gender': gender,
        'tvr': tvr.astype(np.float32)
    })
    
    return df

def generate_events(start_date, end_date):
    """
    Generates a sample events calendar including holidays and sports.
    """
    events = [
        # History (2023)
        {'timestamp': '2023-12-24 20:00:00', 'event_category': 'Christmas_Eve', 'is_historical': True},
        {'timestamp': '2023-12-31 23:00:00', 'event_category': 'New_Year_Eve', 'is_historical': True},
        # History (2024)
        {'timestamp': '2024-07-14 21:00:00', 'event_category': 'Football_Final', 'is_historical': True},
        {'timestamp': '2024-12-24 20:00:00', 'event_category': 'Christmas_Eve', 'is_historical': True},
        {'timestamp': '2024-12-31 23:00:00', 'event_category': 'New_Year_Eve', 'is_historical': True},
        # History (2025)
        {'timestamp': '2025-12-24 20:00:00', 'event_category': 'Christmas_Eve', 'is_historical': True},
        {'timestamp': '2025-12-31 23:00:00', 'event_category': 'New_Year_Eve', 'is_historical': True},
        # Future (2026)
        {'timestamp': '2026-02-08 19:00:00', 'event_category': 'Olympics_Opening', 'is_historical': False},
        {'timestamp': '2026-07-12 21:00:00', 'event_category': 'Football_Final', 'is_historical': False},
        {'timestamp': '2026-12-24 20:00:00', 'event_category': 'Christmas_Eve', 'is_historical': False},
        {'timestamp': '2026-12-31 23:00:00', 'event_category': 'New_Year_Eve', 'is_historical': False},
    ]
    return pd.DataFrame(events)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TV audience data")
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-01-01', help='End date (YYYY-MM-DD) - marks end of history')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    np.random.seed(42) # Reproducibility
    
    print(f"Generating synthetic AGF data from {args.start} to {args.end}...")
    dfs = []
    for demo in REAL_BASE_MINUTES.keys():
        print(f"Generating {demo}...")
        df = generate_demographic_data(demo, args.start, args.end)
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Save historical data
    hist_path = os.path.join(args.output_dir, 'synthetic_history.csv')
    full_df.to_csv(hist_path, index=False)
    print(f"Saved historical data to {hist_path} ({len(full_df)} rows)")
    
    # Generate and save events
    events_df = generate_events(args.start, '2027-01-01')
    events_path = os.path.join(args.output_dir, 'events_calendar.csv')
    events_df.to_csv(events_path, index=False)
    print(f"Saved events calendar to {events_path}")

if __name__ == "__main__":
    main()
