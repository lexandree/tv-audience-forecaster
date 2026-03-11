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

# New imports for alternate engines
from src.models.prophet_forecaster import train_prophet_model, predict_prophet_model
from src.models.convlstm_forecaster import train_convlstm_model, predict_convlstm_model, create_sequences

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # PyTorch seed
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="TV Audience Forecasting CLI")
    parser.add_argument('--input', required=True, help="Path to historical audience CSV")
    parser.add_argument('--events', required=False, help="Path to events calendar CSV")
    parser.add_argument('--output', required=True, help="Path to output forecast CSV")
    parser.add_argument('--start-date', required=True, help="Forecast start date (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, default=365, help="Number of days to forecast")
    parser.add_argument('--engine', choices=['fft', 'prophet', 'convlstm'], default='fft', help="Forecasting engine to use")
    parser.add_argument('--k-freq', type=int, default=100, help="Number of Top-K frequencies to extract (FFT only)")
    
    args = parser.parse_args()
    
    logging.info(f"Loading and processing historical data from {args.input}...")
    historical_segments = process_historical_data(args.input)
    
    events_df = None
    if args.events:
        logging.info(f"Loading events calendar from {args.events}...")
        events_df = load_events_data(args.events)
        
    logging.info(f"Generating future timestamps for {args.days} days starting {args.start_date}...")
    future_dates = generate_future_timestamps(args.start_date, args.days)
    
    # Calculate optional future regressors globally (e.g. Schulferien)
    schulferien_idx = calculate_schulferien_index(future_dates)
    future_events = events_df[events_df['is_historical'] == False] if events_df is not None else pd.DataFrame()
    
    all_forecasts = []

    if args.engine == 'fft':
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
                
        logging.info("Extrapolating and applying impacts...")
        for key, profile in fft_profiles.items():
            extrapolated_signal = reconstruct_from_fft(profile, length=len(future_dates))
            base_forecast_df = pd.DataFrame({
                'predicted_tvr': extrapolated_signal,
                'age_group': key[0],
                'gender': key[1]
            }, index=future_dates)
            
            adjusted_forecast = apply_event_impacts(base_forecast_df, future_events, all_impact_profiles, key)
            all_forecasts.append(adjusted_forecast)

    elif args.engine == 'prophet':
        logging.info("Training Prophet models...")
        for key, df in historical_segments.items():
            logging.info(f"  Training Prophet for {key[0]} {key[1]}...")
            
            model = train_prophet_model(df, regressors=None)
            
            logging.info(f"  Predicting future for {key[0]} {key[1]}...")
            predicted_tvr = predict_prophet_model(model, future_dates, future_regressors=None)
            
            forecast_df = pd.DataFrame({
                'predicted_tvr': predicted_tvr,
                'age_group': key[0],
                'gender': key[1]
            }, index=future_dates)
            all_forecasts.append(forecast_df)

    elif args.engine == 'convlstm':
        logging.info("Training ConvLSTM models...")
        window_size = 168 # 1 week lookback
        for key, df in historical_segments.items():
            logging.info(f"  Training ConvLSTM for {key[0]} {key[1]}...")
            
            # Simple scaling (min-max to [0,1] roughly, or standard scaling)
            # For simplicity, we just divide by max
            max_val = df['tvr'].max()
            if max_val == 0: max_val = 1.0
            
            scaled_data = df['tvr'].values / max_val
            X, y = create_sequences(scaled_data, window_size)
            
            # Very short epochs for demonstration, real model needs more.
            model = train_convlstm_model(X, y, epochs=2, batch_size=256)
            
            logging.info(f"  Predicting future for {key[0]} {key[1]}...")
            # Take the last window from history as the seed
            initial_seq = scaled_data[-window_size:]
            
            # Predict future_steps
            future_steps = len(future_dates)
            scaled_predictions = predict_convlstm_model(model, initial_seq, future_steps)
            
            # Inverse scale
            predicted_tvr = scaled_predictions * max_val
            
            forecast_df = pd.DataFrame({
                'predicted_tvr': predicted_tvr,
                'age_group': key[0],
                'gender': key[1]
            }, index=future_dates)
            all_forecasts.append(forecast_df)
            
    final_forecast_df = pd.concat(all_forecasts)
    
    logging.info(f"Exporting forecast to {args.output}...")
    export_forecast(final_forecast_df, args.output)
    logging.info("Done!")

if __name__ == "__main__":
    main()
