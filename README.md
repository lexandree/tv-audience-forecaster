# tv-audience-forecaster

A time-series forecasting CLI tool designed to predict hourly TV ratings (TVR) segmented by age and gender. Currently, the core engine uses Fast Fourier Transform (FFT) for extracting macroscopic seasonality, overlaid with external event impact modeling (holidays, sports). 

*Note: This project is an active research repository. The FFT approach, Meta's Prophet, and a ConvLSTM model have been implemented and benchmarked on synthetic data. **It is crucial to note that these synthetic datasets, while useful for initial validation, do not possess the nuanced complexities of real-world TV viewership data. Advanced model tuning, especially for deep learning approaches like ConvLSTM, will yield significantly more accurate and representative results when applied to natural, diverse datasets.***

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

## TO DO / Future Work
- [ ] **Real-world Panel Data:** Source high-resolution panel datasets (e.g., from BARC India or open European providers) to expose models to authentic noise and variance, crucial for ConvLSTM optimization.
- [ ] **Advanced Meteorological Regressors:** Extend weather data via the Open-Meteo API (e.g., sunshine duration, wind speed) to measure nuanced impacts on viewing behavior.
- [ ] **Commercial Demographics:** Expand the target variable segmentations to focus on traditional commercial demographics like 14-49.
- [ ] **Hyperparameter Tuning:** Conduct a comprehensive grid search and implement automated learning rate scheduling for the PyTorch ConvLSTM.
