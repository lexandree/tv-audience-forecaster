import argparse
import sys
import pandas as pd
import numpy as np
import random
import logging
from src.data.pipeline import process_historical_data
from src.data.ingestion import load_events_data
from src.data.events_ingestion import map_events_to_timeseries
from src.data.holidays import calculate_schulferien_index
from src.models.pipeline import train_fft_models, evaluate_models
from src.models.residuals import calculate_residuals
from src.models.event_impact import build_impact_profile
from src.models.forecaster import generate_future_timestamps, apply_event_impacts
from src.models.fft_reconstructor import reconstruct_from_fft
from src.data.export import export_forecast

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="TV Audience Forecasting CLI")
    parser.add_argument('--input', required=True, help="Path to historical audience CSV")
    parser.add_argument('--events', required=False, help="Path to events calendar CSV")
    parser.add_argument('--output', required=True, help="Path to output forecast CSV")
    parser.add_argument('--start-date', required=True, help="Forecast start date (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, default=365, help="Number of days to forecast")
    parser.add_argument('--k-freq', type=int, default=100, help="Number of Top-K frequencies to extract")
    
    args = parser.parse_args()
    
    logging.info(f"Loading and processing historical data from {args.input}...")
    historical_segments = process_historical_data(args.input)
    
    events_df = None
    if args.events:
        logging.info(f"Loading events calendar from {args.events}...")
        events_df = load_events_data(args.events)
        
    logging.info(f"Training FFT models (Top-{args.k_freq} frequencies)...")
    fft_profiles = train_fft_models(historical_segments, k=args.k_freq)
    
    # Calculate baseline errors
    mape_scores = evaluate_models(historical_segments, fft_profiles)
    for key, score in mape_scores.items():
        logging.info(f"  {key[0]} {key[1]} - Baseline MAPE: {score*100:.2f}%")
        
    # Calculate Event Impacts
    all_impact_profiles = []
    if events_df is not None:
        logging.info("Calculating historical event impacts...")
        hist_events = events_df[events_df['is_historical'] == True]
        
        for key, df in historical_segments.items():
            profile = fft_profiles[key]
            reconstructed = reconstruct_from_fft(profile, length=len(df))
            residuals_df = calculate_residuals(df, reconstructed)
            mapped_residuals = map_events_to_timeseries(residuals_df, hist_events)
            
            profiles = build_impact_profile(mapped_residuals, key)
            all_impact_profiles.extend(profiles)
            
    logging.info(f"Generating forecast for {args.days} days starting {args.start_date}...")
    future_dates = generate_future_timestamps(args.start_date, args.days)
    
    # Pre-calculate Schulferien Index (optional, but requested for the model)
    schulferien_idx = calculate_schulferien_index(future_dates)
    
    future_events = events_df[events_df['is_historical'] == False] if events_df is not None else pd.DataFrame()
    
    all_forecasts = []
    for key, profile in fft_profiles.items():
        # 1. Base FFT extrapolation
        extrapolated_signal = reconstruct_from_fft(profile, length=len(future_dates))
        
        base_forecast_df = pd.DataFrame({
            'predicted_tvr': extrapolated_signal,
            'age_group': key[0],
            'gender': key[1]
        }, index=future_dates)
        
        # 2. Apply explicit event adjustments
        adjusted_forecast = apply_event_impacts(base_forecast_df, future_events, all_impact_profiles, key)
        
        all_forecasts.append(adjusted_forecast)
        
    final_forecast_df = pd.concat(all_forecasts)
    
    logging.info(f"Exporting forecast to {args.output}...")
    export_forecast(final_forecast_df, args.output)
    logging.info("Done!")

if __name__ == "__main__":
    main()
