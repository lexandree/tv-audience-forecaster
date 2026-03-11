import pandas as pd
import logging
from src.data.pipeline import process_historical_data
from src.models.pipeline import train_fft_models, evaluate_models
from src.models.prophet_forecaster import train_prophet_model, predict_prophet_model
from src.models.convlstm_forecaster import train_convlstm_model, predict_convlstm_model, create_sequences
from src.evaluation.metrics import calculate_mape
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_benchmark():
    logging.info("=== Starting Model Benchmarking ===")
    
    # Use synthetic data for benchmarking
    # We will hold out the last 30 days (720 hours) for testing
    df_path = "data/raw/synthetic_history.csv"
    logging.info(f"Loading data from {df_path}")
    all_segments = process_historical_data(df_path)
    
    results = []
    
    for key, df in all_segments.items():
        logging.info(f"\n--- Benchmarking Demographic: {key[0]} {key[1]} ---")
        
        # Train/Test Split (Hold out last 30 days)
        test_hours = 24 * 30
        train_df = df.iloc[:-test_hours]
        test_df = df.iloc[-test_hours:]
        test_actuals = test_df['tvr'].values
        future_dates = test_df.index
        
        # 1. FFT
        start_time = time.time()
        # Train FFT only on train_df
        # Note: train_fft_models takes a dict, so we wrap it
        fft_profiles = train_fft_models({key: train_df}, k=100)
        profile = fft_profiles[key]
        
        from src.models.fft_reconstructor import reconstruct_from_fft
        # To predict future, we reconstruct the full length and take the end
        full_extrapolated = reconstruct_from_fft(profile, length=len(df))
        fft_predictions = full_extrapolated[-test_hours:]
        
        fft_time = time.time() - start_time
        fft_mape = calculate_mape(test_actuals, fft_predictions)
        
        results.append({
            'Demographic': f"{key[0]} {key[1]}",
            'Model': 'FFT (Top-100)',
            'MAPE': fft_mape,
            'Time_Seconds': fft_time
        })
        logging.info(f"FFT -> MAPE: {fft_mape*100:.2f}%, Time: {fft_time:.2f}s")
        
        # 2. Prophet
        start_time = time.time()
        prophet_model = train_prophet_model(train_df)
        prophet_preds = predict_prophet_model(prophet_model, future_dates)
        prophet_time = time.time() - start_time
        prophet_mape = calculate_mape(test_actuals, prophet_preds)
        
        results.append({
            'Demographic': f"{key[0]} {key[1]}",
            'Model': 'Prophet',
            'MAPE': prophet_mape,
            'Time_Seconds': prophet_time
        })
        logging.info(f"Prophet -> MAPE: {prophet_mape*100:.2f}%, Time: {prophet_time:.2f}s")
        
        # 3. ConvLSTM
        start_time = time.time()
        window_size = 168
        max_val = train_df['tvr'].max() if train_df['tvr'].max() > 0 else 1.0
        scaled_train = train_df['tvr'].values / max_val
        X, y = create_sequences(scaled_train, window_size)
        
        # Extremely fast mock training for benchmark script (in reality this takes much longer)
        convlstm_model = train_convlstm_model(X, y, epochs=1, batch_size=256)
        
        initial_seq = scaled_train[-window_size:]
        scaled_preds = predict_convlstm_model(convlstm_model, initial_seq, test_hours)
        convlstm_preds = scaled_preds * max_val
        
        convlstm_time = time.time() - start_time
        convlstm_mape = calculate_mape(test_actuals, convlstm_preds)
        
        results.append({
            'Demographic': f"{key[0]} {key[1]}",
            'Model': 'ConvLSTM',
            'MAPE': convlstm_mape,
            'Time_Seconds': convlstm_time
        })
        logging.info(f"ConvLSTM -> MAPE: {convlstm_mape*100:.2f}%, Time: {convlstm_time:.2f}s")
        
    # Summarize
    results_df = pd.DataFrame(results)
    print("\n=== BENCHMARK SUMMARY ===")
    print(results_df.groupby('Model')[['MAPE', 'Time_Seconds']].mean().sort_values('MAPE'))
    
if __name__ == "__main__":
    run_benchmark()
