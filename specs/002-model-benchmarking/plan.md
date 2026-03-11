# Implementation Plan: Model Benchmarking (Prophet & ConvLSTM)

**Branch**: `002-model-benchmarking` | **Date**: 2026-03-11 | **Spec**: [spec.md](./spec.md)

## Summary
This phase introduces two new predictive engines to the `tv-audience-forecaster` repository: Meta's **Prophet** and a **ConvLSTM** neural network. The existing FFT methodology will serve as the baseline. 

Prophet will naturally accept our external regressors (events, weather, holidays). ConvLSTM will require preparing the data into a 3D tensor format (samples, time_steps, features). A new command-line interface argument will govern which model runs, and a dedicated benchmarking script will run all three to compile a comparison matrix.

## Technical Context
**Dependencies to add**: 
- `prophet` (for Meta Prophet)
- `torch` or `tensorflow` (for ConvLSTM)
- `scikit-learn` (for data scaling necessary for Neural Networks)

**Architecture Additions**:
- `src/models/prophet_forecaster.py`
- `src/models/convlstm_forecaster.py`
- `src/evaluation/benchmark.py`

## Project Structure Changes
```text
src/
├── models/
│   ├── prophet_forecaster.py  # New
│   ├── convlstm_forecaster.py # New
│   └── ...
├── evaluation/
│   ├── benchmark.py           # New: runs comparison matrix
│   └── ...
```

## Constitution Check
- [x] **Test-First**: Are deterministic tests planned? (Yes, mocking external models to test integration).
- [x] **Simplicity**: Are we over-engineering? (Using standard library implementations of Prophet and simple PyTorch/Keras ConvLSTM structures without excessive custom layers).
