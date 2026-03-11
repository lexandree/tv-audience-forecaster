# Benchmark Report: Forecasting Engines

**Date**: 2026-03-11  
**Dataset**: `synthetic_history.csv` (Calibrated to German AGF 2023-2025)  
**Test Setup**: Hold-out the last 30 days (720 hours).  

## Summary of Results

| Model | Average MAPE | Average Time (Seconds) | Notes |
|-------|-------------:|-----------------------:|-------|
| **Prophet** | 9.47% | 8.72s | **Best Accuracy**. Handles the multiple seasonalities effortlessly. Moderate training time. |
| **FFT (Top-100)** | 17.08% | 0.07s | **Fastest**. Extremely lightweight, but suffers slightly when predicting out of sample without the external regressors directly embedded. |
| **ConvLSTM** | 33.35% | 17.28s | **Requires Tuning**. The 1-epoch benchmark shows it needs significant tuning (epochs, batch size, deeper architecture) to converge, making it expensive to train. |

## Conclusion
For production forecasting of TV audiences where computational resources allow ~10 seconds per demographic, **Prophet** is the clear winner in terms of out-of-the-box accuracy and ease of integrating external regressors like the Schulferien index or Weather data. 

If ultra-low latency is required (e.g., real-time continuous updates across thousands of micro-segments), the **FFT** model provides a solid baseline almost instantaneously.
