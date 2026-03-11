---
description: "Task list for Model Benchmarking implementation"
---

# Tasks: Model Benchmarking

## Phase 1: Environment & Tooling Update
- [x] T001 Add `prophet`, `torch` (or tensorflow), and `scikit-learn` to `pyproject.toml` dependencies.
- [x] T002 Update virtual environment with new dependencies.

## Phase 2: Prophet Implementation
- [x] T003 Write unit tests for Prophet integration in `tests/unit/test_prophet.py` (mocking the fit/predict).
- [x] T004 Implement `src/models/prophet_forecaster.py`, adapting our `DemographicTimeSeries` into the expected `ds`, `y` Prophet format.
- [x] T005 Integrate external regressors (holidays, weather) into the Prophet model configuration using `add_regressor`.

## Phase 3: ConvLSTM Implementation
- [x] T006 Write unit tests for data windowing logic required for ConvLSTM in `tests/unit/test_windowing.py`.
- [x] T007 Implement data scaler and windowing functions (e.g., creating 168-hour lookback sequences) in `src/models/convlstm_forecaster.py`.
- [x] T008 Define the ConvLSTM architecture (PyTorch/Keras) and implement training loop with early stopping.
- [x] T009 Implement auto-regressive inference loop for 1-year extrapolation.

## Phase 4: CLI & Benchmarking Integration
- [x] T010 Update `src/cli/main.py` with an `--engine` argument (choices: `fft`, `prophet`, `convlstm`).
- [x] T011 Create `src/evaluation/benchmark.py` to run a comparative backtest on a hold-out set (e.g., last 3 months of history) across all three models.
- [x] T012 Run the benchmark and generate the final analysis report.

