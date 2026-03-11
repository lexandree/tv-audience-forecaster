# Feature Specification: Model Benchmarking (Prophet & ConvLSTM)

**Feature Branch**: `002-model-benchmarking`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User request to benchmark the existing FFT baseline against Prophet and ConvLSTM.

## 1. Problem Statement
The current implementation of `tv-audience-forecaster` uses an FFT-based method to extract macro-seasonalities and additive residual models for external events. While lightweight and fast, we need to rigorously evaluate if modern forecasting models—specifically Meta's Prophet (a robust generalized additive model) and ConvLSTM (a deep learning approach for capturing spatio-temporal dependencies)—can achieve better accuracy (lower MAPE) when forecasting TV audiences, particularly under the influence of complex external factors (Schulferien, weather).

## 2. Target Audience / Use Case
- **Data Scientists / Forecasters**: Need empirical evidence to choose the optimal production algorithm based on the trade-off between computational cost, ease of maintenance, and MAPE.
- **Business Stakeholders**: Need the most accurate predictions of TVR to optimize ad-inventory pricing.

## 3. Scope
**In Scope**:
- Implementation of a `ProphetForecaster` module integrating with the existing data pipeline.
- Implementation of a `ConvLSTMForecaster` module using PyTorch or TensorFlow/Keras.
- Expansion of the CLI to allow selecting the forecasting engine (`--engine fft|prophet|convlstm`).
- A unified evaluation module that outputs a benchmarking report comparing the three models on the synthetic German AGF dataset.

**Out of Scope**:
- Deployment of models to production (this is strictly an evaluation phase).
- Hyperparameter grid-search taking more than a few hours (we will use sensible defaults/heuristics to prove the concept).

## 4. Key Metrics (Success Criteria)
- **MAPE Comparison**: Generate a comprehensive report comparing the Mean Absolute Percentage Error of FFT, Prophet, and ConvLSTM across all demographics (`18-24 M`, `55+ F`, etc.).
- **Performance Profiling**: Track and compare training time and inference time for a 1-year forecast.
- **SC-001**: The CLI successfully executes an end-to-end forecast for all three engines using the same input data.
- **SC-002**: A final benchmarking report is produced (CSV or markdown) detailing the metrics above.
