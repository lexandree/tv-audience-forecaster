import pandas as pd
import logging
from src.data.pipeline import process_historical_data
from src.data.ingestion import load_events_data
from src.data.events_ingestion import map_events_to_timeseries
from src.data.holidays import calculate_schulferien_index
from src.models.pipeline import train_fft_models
from src.models.prophet_forecaster import train_prophet_model, predict_prophet_model
from src.models.convlstm_forecaster import train_convlstm_model, predict_convlstm_model, create_sequences
from src.evaluation.metrics import calculate_mape
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')

def prepare_regressors(df: pd.DataFrame, all_events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a dataframe of external regressors aligned with the main dataframe's index.
    Includes Schulferien index and one-hot encoded event categories.
    """
    # 1. Schulferien Index
    regressors = pd.DataFrame(index=df.index)
    regressors['schulferien'] = calculate_schulferien_index(df.index)

    # 2. Events
    if all_events_df is not None:
        mapped = map_events_to_timeseries(regressors, all_events_df)

        # One-hot encode event categories
        if not mapped['event_category'].isna().all():
            dummies = pd.get_dummies(mapped['event_category'], prefix='event')
            regressors = regressors.join(dummies)

    return regressors.fillna(0)


def run_benchmark():
    logging.info("=== Starting Model Benchmarking ===")

    df_path = "data/raw/synthetic_history.csv"
    events_path = "data/raw/events_calendar.csv"
    logging.info(f"Loading data from {df_path}")

    all_segments = process_historical_data(df_path)
    events_df = load_events_data(events_path)

    results = []

    for key, df in all_segments.items():
        logging.info(f"\n--- Benchmarking Demographic: {key[0]} {key[1]} ---")

        test_hours = 24 * 30
        train_df = df.iloc[:-test_hours]
        test_df = df.iloc[-test_hours:]
        test_actuals = test_df['tvr'].values
        future_dates = test_df.index

        # --- FFT (Baseline) ---
        start_time = time.time()
        fft_profiles = train_fft_models({key: train_df}, k=100)
        profile = fft_profiles[key]
        from src.models.fft_reconstructor import reconstruct_from_fft
        full_extrapolated = reconstruct_from_fft(profile, length=len(train_df) + test_hours)
        fft_predictions = full_extrapolated[-test_hours:]
        fft_time = time.time() - start_time
        fft_mape = calculate_mape(test_actuals, fft_predictions)
        results.append({'Demographic': f"{key[0]} {key[1]}",'Model': 'FFT (Top-100)','MAPE': fft_mape,'Time_Seconds': fft_time})
        logging.info(f"FFT -> MAPE: {fft_mape*100:.2f}%, Time: {fft_time:.2f}s")

        # --- Prophet (Base) ---
        start_time = time.time()
        prophet_model = train_prophet_model(train_df)
        prophet_preds = predict_prophet_model(prophet_model, future_dates)
        prophet_time = time.time() - start_time
        prophet_mape = calculate_mape(test_actuals, prophet_preds)
        results.append({'Demographic': f"{key[0]} {key[1]}",'Model': 'Prophet (Base)','MAPE': prophet_mape,'Time_Seconds': prophet_time})
        logging.info(f"Prophet (Base) -> MAPE: {prophet_mape*100:.2f}%, Time: {prophet_time:.2f}s")

        # --- Prophet (with Regressors) ---
        start_time = time.time()
        train_regressors = prepare_regressors(train_df, events_df[events_df['is_historical']])
        test_regressors = prepare_regressors(test_df, events_df[~events_df['is_historical']])

        # Ensure all regressor columns exist in both dataframes
        all_cols = train_regressors.columns.union(test_regressors.columns)
        train_regressors = train_regressors.reindex(columns=all_cols, fill_value=0)
        test_regressors = test_regressors.reindex(columns=all_cols, fill_value=0)

        prophet_reg_model = train_prophet_model(train_df, regressors=train_regressors)
        prophet_reg_preds = predict_prophet_model(prophet_reg_model, future_dates, future_regressors=test_regressors)
        prophet_reg_time = time.time() - start_time
        prophet_reg_mape = calculate_mape(test_actuals, prophet_reg_preds)
        results.append({'Demographic': f"{key[0]} {key[1]}",'Model': 'Prophet (Regressors)','MAPE': prophet_reg_mape,'Time_Seconds': prophet_reg_time})
        logging.info(f"Prophet (Regressors) -> MAPE: {prophet_reg_mape*100:.2f}%, Time: {prophet_reg_time:.2f}s")

    results_df = pd.DataFrame(results)
    print("\n=== BENCHMARK SUMMARY ===")
    summary = results_df.groupby('Model')[['MAPE', 'Time_Seconds']].mean().sort_values('MAPE')
    summary['MAPE'] = summary['MAPE'].apply(lambda x: f"{x*100:.2f}%")
    print(summary)

if __name__ == "__main__":
    run_benchmark()

