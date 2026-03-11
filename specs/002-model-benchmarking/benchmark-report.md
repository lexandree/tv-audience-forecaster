# Benchmark Report: Forecasting Engines

**Date**: 2026-03-11  
**Dataset**: `synthetic_history.csv` (Calibrated to German AGF 2023-2025)  
**Test Setup**: Hold-out the last 30 days (720 hours).  

*Note on Data: The synthetic dataset used for this benchmark is simplified and designed for initial validation. It may not fully capture the complexities and noise inherent in real-world TV viewership data. Therefore, advanced models like ConvLSTM, which typically benefit from richer, more diverse datasets, might show improved performance and more representative results when tuned on natural data.* 

## Summary of Results

| Model | Average MAPE | Average Time (Seconds) | Notes |
|-------|-------------:|-----------------------:|-------|
| **Prophet (Base)** | 9.47% | 8.80s | **Best Accuracy**. Handles the multiple seasonalities effortlessly. Moderate training time. |
| **Prophet (Regressors)** | 9.48% | 10.80s | Slight performance degradation on this synthetic dataset; may perform better on real data. |
| **FFT (Top-100)** | 17.09% | 0.08s | **Fastest**. Extremely lightweight, but suffers slightly when predicting out of sample without the external regressors directly embedded. |
| **ConvLSTM** | 33.35% | 17.28s | **Requires Tuning**. The 1-epoch benchmark shows it needs significant tuning (epochs, batch size, deeper architecture) to converge, making it expensive to train. |

## Conclusion
For production forecasting of TV audiences where computational resources allow ~10 seconds per demographic, **Prophet (Base)** is the clear winner in terms of out-of-the-box accuracy and ease of integration for this dataset. 

If ultra-low latency is required (e.g., real-time continuous updates across thousands of micro-segments), the **FFT** model provides a solid baseline almost instantaneously.

Moving forward, further development and hyperparameter tuning of all models, especially ConvLSTM, should be conducted on more natural, diverse, and larger datasets to accurately assess their full potential in real-world scenarios.
