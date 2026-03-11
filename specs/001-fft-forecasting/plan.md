# Implementation Plan: TV Audience Forecasting by Age/Gender

**Branch**: `001-fft-forecasting` | **Date**: 2026-03-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-fft-forecasting/spec.md`

## Summary

This project implements a TV audience forecasting system based on the Fast Fourier Transform (FFT). Because raw commercial TV data (e.g., AGF, BARB) is strictly under NDA, this project is structured as a **multi-model portfolio**, proving the methodology across different datasets and markets:
1. **German Market Model**: Utilizing synthetic hourly data calibrated to real AGF Videoforschung 2023–2025 base minutes, overlaid with German Schulferien (school holidays) indices and Open-Meteo weather adjustments.
2. **European Model**: Utilizing the granular `carrenyo` GitHub device dataset (minute/hourly) for pure high-resolution FFT extraction.
3. **UK/India/Streaming Models**: Utilizing aggregated open datasets (weekly BARB, BARC India TRP ratings, Netflix user behavior) to benchmark FFT against models like Prophet and ConvLSTM.

The core system ingests historical time-series data, segments it by demographics, applies FFT to extract top-K seasonal frequencies, and extrapolates a full-year forecast, factoring in external events.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: pandas (data manipulation), scipy (FFT implementation), numpy
**Storage**: CSV / Excel files for input data and output forecasts
**Testing**: pytest (strictly following TDD)
**Target Platform**: CLI / Local execution / Jupyter Notebooks for EDA
**Project Type**: Data Processing CLI / Library / Portfolio Repository
**Performance Goals**: Process historical data and generate 1-year forecast for all configured demographic groups in < 5 minutes
**Constraints**: High memory efficiency required for handling hourly demographic datasets (use proper Pandas dtypes like float32 or categorical types).
**Scale/Scope**: Minimum 3 years of historical input data (where available) or 52 weeks (minimum for FFT yearly extraction); outputting predictions at the same resolution as the input (e.g., 8,760 predictions for hourly data).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **FFT-Based Forecasting**: Does the approach utilize FFT for yearly predictions? (Yes, extracting Top-K frequencies)
- [x] **Demographic Granularity**: Is the data strictly segmented by age and gender? (Yes, pipeline is strictly segmented where data supports it)
- [x] **Test-First**: Are deterministic unit tests planned for the mathematical operations? (Yes, via pytest for all math logic)
- [x] **Historical Reconstruction**: Is there a validation step against historical baseline data? (Yes, comparing against SPLY baseline)
- [x] **Simplicity & Observability**: Is the FFT pipeline observable and as simple as possible? (Yes, using standard scipy FFT without overly complex deep learning layers)

## Project Structure

### Documentation (this feature)

```text
specs/001-fft-forecasting/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── interface.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── data/          # Ingestion, segmentation, and seasonal interpolation
├── models/        # FFT extraction and reconstruction logic
├── evaluation/    # Baseline comparisons (SPLY) and MAPE calculation
├── cli/           # Command-line interface and configuration
└── utils/         # Observability and logging

scripts/
└── generate_synthetic_agf.py # Calibration script for German market data

tests/
├── unit/          # Deterministic tests for FFT math and data transforms
├── integration/   # Pipeline flow tests
└── contract/      # Input/output format validations
```

**Structure Decision**: Option 1 (Single project) tailored for a Python Data CLI, with scripts for synthetic data generation.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
