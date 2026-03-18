---
title: "TV Audience Forecaster"
description: "A time-series forecasting pipeline predicting hourly TV viewership by demographic. Extracted macro-seasonality with FFT, built an external event modeling layer, and rigorously benchmarked against Meta Prophet and ConvLSTM on a simulated AGF dataset."
pubDate: 2026-03-18
tags: ["Time-Series", "Prophet", "FFT", "ConvLSTM", "Data Science", "Python"]
githubUrl: "https://github.com/lexandree/tv-audience-forecaster"
featured: true
order: 9
---

## Project Overview

Television audience ratings (TVR) display complex, overlapping periodicities—daily primetime peaks, weekend effects, and seasonal trends—disrupted heavily by external events like holidays and live sports. Due to strict NDAs on commercial hourly TV data (e.g., AGF, BARB), predicting these micro-segments is difficult for external researchers.

This project implements a high-performance **time-series forecasting pipeline** that models demographic viewership by isolating seasonal frequencies and applying regressive adjustments. 

![Forecast Plot for 2026](https://raw.githubusercontent.com/lexandree/tv-audience-forecaster/master/docs/forecast_plot.png)

## Key Technical Decisions & Methodology

### 1. The Multi-Model Benchmark
I chose to build the system as a benchmarking portfolio to prove which architecture balances latency, accuracy, and ease of maintenance:
* **Fast Fourier Transform (FFT)**: The baseline. Used `scipy.fft` to extract Top-K frequencies. While incredibly fast (<0.1s for 8,760 hours), it requires a separate additive residual model to adjust for sudden events.
* **Meta Prophet**: Handled the multiple seasonalities effortlessly and natively supported external regressors.
* **ConvLSTM**: A deep learning approach (using PyTorch) built to capture spatio-temporal dependencies.

### 2. External Factors Integration
A simple binary "is_holiday" flag is insufficient for countries like Germany, where school holidays are staggered by state. 
* I calculated a continuous, **population-weighted Schulferien index** based on the exact school holiday dates across the 16 German federal states using Destatis population weights.
* Integrated **Open-Meteo API** (DWD ICON forecasts) to generate weather-based multipliers (e.g., extreme heat >28°C or heavy rain >10mm increasing indoor audience).

### 3. Data Simulation
Because raw hourly commercial data is closed, I built a Python generator (`scripts/generate_synthetic_agf.py`) that outputs synthetic data calibrated exactly to real AGF Videoforschung 2023–2025 base minutes. This ensured my models had realistic variance, noise, and seasonal shapes to train on.

## Benchmarking Results
On a 30-day hold-out test set:

| Model | Average MAPE | Average Time (Seconds) |
|-------|-------------:|-----------------------:|
| **Prophet (Base)** | 9.47% | 8.80s |
| **FFT (Top-100)** | 17.09% | 0.08s |
| **ConvLSTM** | 33.35% | 17.28s |

**Prophet** proved to be the most accurate for this dataset, easily capturing overlapping seasonalities without overfitting. **FFT** remains a viable, hyper-fast option if computational resources are constrained (e.g., thousands of segments predicted continuously).

## Future Work
- Validate the models against natural, high-resolution panel datasets (e.g., GitHub open European datasets or Kaggle BARC India data) to tune the ConvLSTM architecture.
- Expand external regressors with more granular meteorological data.
