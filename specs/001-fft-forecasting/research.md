# Research & Technical Decisions: 001-fft-forecasting

## Data Sources Strategy
Due to the strict NDA and commercial nature of raw, hourly TV audience data by age and gender (e.g., AGF in Germany, BARB in UK, BARC in India), this project adopts a multi-dataset portfolio approach to validate the forecasting models:
1. **German Market (Synthetic & Calibrated)**: Since full raw AGF data is closed, we will generate synthetic hourly data calibrated to real AGF Videoforschung 2023–2025 averages (e.g., base minutes per age/gender group). This dataset will uniquely incorporate our `Schulferien` index and Open-Meteo weather adjustments.
2. **European Hourly Data (GitHub `carrenyo`)**: For the pure, granular hourly FFT proof-of-concept, we will utilize the open-source `TV-Viewer-Demographics-Machine-Learning` dataset (350k+ devices, 52 weeks, minute/hourly aggregation with age/gender). This is the best candidate for extracting the 1/24h and 1/168h frequencies.
3. **UK (BARB) / India (BARC) / Streaming (Kaggle)**: We will test the FFT methodology and model comparisons (Prophet, ConvLSTM) against real aggregated open datasets, such as weekly BARB reports, Kaggle's BARC TRP ratings for India, or Netflix user behavior datasets, to demonstrate versatility across different resolutions and markets.

## FFT Implementation Library
- **Decision**: Use `scipy.fft` module.
- **Rationale**: While `numpy.fft` is available, `scipy.fft` provides a more comprehensive and highly optimized set of tools for Fast Fourier Transforms, often with better performance backends. It is the standard for scientific computing in Python.
- **Alternatives considered**: `numpy.fft` (slightly fewer features/optimizations).

## Missing Data Handling (Seasonal Interpolation)
- **Decision**: Implement a custom Pandas transformer using `.shift()` and `.groupby()`.
- **Rationale**: Standard `interpolate(method='linear')` destroys the periodic nature of TV audiences. By shifting the data by exactly one week (`168` hours) or grouping by `(day_of_week, hour)` to find the historical average, we preserve the signal's shape before feeding it to FFT.
- **Alternatives considered**: Simple Linear Interpolation (rejected due to signal distortion). Zero-padding (rejected due to artificial frequency artifacts).

## Frequency Selection (Top-K)
- **Decision**: Extract amplitudes from the FFT output, sort descending, and truncate all but the top `K` frequencies (where `K` is a configurable parameter, default e.g., 100).
- **Rationale**: TV audience data contains significant high-frequency noise. Retaining only the highest amplitude frequencies acts as a natural low-pass/band-pass filter, capturing the true underlying macro-seasonality (daily, weekly, yearly cycles).
- **Alternatives considered**: Low-pass cutoff filter (harder to tune than simply picking the strongest signals). Full spectrum (rejected as it overfits to noise).

## External Events Impact Modeling
- **Decision**: Additive Residual Modeling (ARIMAX-style but simpler).
- **Rationale**: FFT perfectly captures baseline seasonality. Events (holidays, sports) are distinct, a-periodic spikes. By subtracting the FFT reconstruction from the actual historical data, we get "Residuals". We can group these residuals by `Event Category` to find the average TVR impact (e.g., "+2.5 TVR for a Football Final"). We then add this expected impact back into the future FFT forecast for scheduled future events.
- **Alternatives considered**: Including events as dummy variables in a heavy ML model (rejected to maintain the purity and speed of the FFT approach).

## Validation Baseline (SPLY)
- **Decision**: "Same Period Last Year" (SPLY) baseline will be calculated by shifting the validation set back by exactly 52 weeks (364 days).
- **Rationale**: 364 days aligns exactly with the day of the week, which is critical for TV programming (e.g., comparing a Sunday to a Sunday).
- **Alternatives considered**: 365 days shift (rejected because it misaligns days of the week, comparing a Monday to a Sunday).

## Future Research: Model Comparisons
- **Task**: Compare the current FFT-based approach with other modern forecasting methods.
- **Models to Evaluate**: 
  - **Prophet**: Developed by Meta, known for handling holidays and strong multiple seasonalities well.
  - **ConvLSTM**: A deep learning approach capable of capturing complex temporal dependencies.
  - **Other Modern Methods**: Investigate recent state-of-the-art time-series models (e.g., N-BEATS, Temporal Fusion Transformers, or Transformer-based models like Informer/Autoformer).
- **Goal**: Establish benchmarks for accuracy (e.g., MAPE), computational efficiency, and ease of integrating external events compared to the FFT baseline.

## External Factors Integration (Weather & School Holidays)
- **School Holidays (Schulferien)**:
  - **Decision**: Calculate a continuous, population-weighted index (0.0 to 1.0) based on the exact school holiday dates across the 16 German federal states (Länder).
  - **Rationale**: TV viewership behaviors change significantly during school holidays. A simple binary "is_holiday" flag is insufficient because holidays in Germany are staggered by state. Weighting by state population (Destatis) provides a highly accurate representation of the national audience impact.
- **Weather Adjustments**:
  - **Decision**: Utilize DWD ICON forecasts via the Open-Meteo API for short-term (up to 14 days) weather-based audience adjustments.
  - **Rationale**: Extreme weather conditions significantly impact TV viewing (e.g., heatwaves >28°C or heavy rain >10mm increase indoor audience by ~9%, extreme cold <-5°C by ~6%). These multipliers will be applied to the base FFT forecast.
