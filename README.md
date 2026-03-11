# tv-audience-forecaster

A time-series forecasting CLI tool designed to predict hourly TV ratings (TVR) segmented by age and gender. Currently, the core engine uses Fast Fourier Transform (FFT) for extracting macroscopic seasonality, overlaid with external event impact modeling (holidays, sports). 

*Note: This project is an active research repository. The FFT approach is currently being benchmarked against Meta's Prophet and ConvLSTM models to determine the optimal production strategy.*

## Quickstart

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Generate Synthetic Training Data (Optional)
If you do not have raw historical data (e.g., AGF, BARB), you can generate a calibrated synthetic dataset for the German market:
```bash
python scripts/generate_synthetic_agf.py --start 2023-01-01 --end 2026-01-01 --output_dir data/raw
```

### 3. Run the Forecaster
Generate a 365-day forecast starting from a specific date:
```bash
tv-forecast \
  --input data/raw/synthetic_history.csv \
  --events data/raw/events_calendar.csv \
  --output data/output/forecast_2026.csv \
  --start-date 2026-01-01 \
  --days 365
```

## How It Works (FFT Baseline)
1. **Ingestion & Segmentation**: Data is strictly segmented by `age_group` and `gender`. Missing values are interpolated using a robust season-over-season fill.
2. **FFT Extraction**: Uses `scipy.fft` to break the historical time-series into its underlying frequency components (daily, weekly, yearly). The top K frequencies are kept to form the macro-seasonal baseline.
3. **Event Impact Residuals**: The difference between actuals and the FFT reconstruction is mapped to known historical events (e.g., "Football Final", Schulferien Index, Weather).
4. **Forecasting**: The baseline is extrapolated into the future, and expected event impacts are added on top of the base waves to yield the final predictions.

## Roadmap & Benchmarking
- [x] Phase 1-6: Implement end-to-end FFT baseline pipeline.
- [x] Implement Meta `Prophet` forecasting engine.
- [x] Implement `ConvLSTM` deep learning engine.
- [x] Compare models on computational efficiency, MAPE, and ease of integrating external regressors (Weather/Holidays).

*Current results show that **Prophet** provides the best accuracy for complex TVR patterns, while **FFT** is the most efficient for large-scale, low-latency processing.*
