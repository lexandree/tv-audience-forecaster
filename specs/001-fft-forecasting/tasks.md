---
description: "Task list for TV Audience Forecasting implementation"
---

# Tasks: TV Audience Forecasting by Age/Gender

**Input**: Design documents from `/specs/001-fft-forecasting/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/interface.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story in accordance with Test-Driven Development (TDD).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Path Conventions: `src/` and `tests/` at repository root.

---

## Phase 0: Data Acquisition & Generation (New)

**Purpose**: Prepare datasets since raw commercial data is closed.

- [x] T000a [P] Generate synthetic hourly dataset for the German market, calibrated using AGF 2023-2025 base minutes in `scripts/generate_synthetic_agf.py`.
- [x] T000b [P] Download/prepare the `carrenyo` European minute/hourly device dataset for core FFT testing.
- [x] T000c [P] Download/prepare Kaggle BARC India dataset for cross-market validation.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure (`src/`, `tests/`, `data/`, `output/`)
- [x] T002 Initialize Python project (e.g., `pyproject.toml` or `requirements.txt`) with pandas, scipy, numpy, pytest
- [x] T003 [P] Configure pytest for testing in `pytest.ini`
- [x] T004 [P] Setup logging utilities in `src/utils/logger.py` to satisfy Observability principle

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T005 Create data model definitions (dataclasses/types) in `src/models/types.py` for AudienceObservation, DemographicTimeSeries, EventCalendar
- [x] T006 [P] Create custom exception classes in `src/utils/exceptions.py` for data and math errors
- [x] T007 Define input/output/events CSV contract validation logic in `src/data/validators.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Historical Data Ingestion and Segmentation (Priority: P1) 🎯 MVP

**Goal**: Ingest historical TV audience data, validate against contract, segment by demographics, and handle missing values via seasonal interpolation.

### Tests for User Story 1

- [x] T008 [P] [US1] Write test for CSV ingestion and contract validation in `tests/contract/test_ingestion.py`
- [x] T009 [P] [US1] Write unit test for demographic segmentation logic in `tests/unit/test_segmentation.py`
- [x] T010 [P] [US1] Write unit test for seasonal interpolation (gap filling) in `tests/unit/test_interpolation.py`

### Implementation for User Story 1

- [x] T011 [P] [US1] Implement CSV reader and data parser in `src/data/ingestion.py`
- [x] T012 [P] [US1] Implement demographic segmentation grouping in `src/data/segmentation.py`
- [x] T013 [US1] Implement seasonal interpolation using `.shift()` and `.groupby()` in `src/data/interpolation.py`
- [x] T014 [US1] Integrate ingestion, segmentation, and interpolation into a single data pipeline in `src/data/pipeline.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - FFT Model Reconstruction and Training (Priority: P1)

**Goal**: Apply FFT to the segmented historical data, extract Top-K seasonal frequencies, and validate reconstruction against SPLY baseline.

### Tests for User Story 2

- [x] T015 [P] [US2] Write deterministic unit test for FFT extraction and Top-K filtering in `tests/unit/test_fft_extraction.py`
- [x] T016 [P] [US2] Write deterministic unit test for time-series reconstruction from frequencies in `tests/unit/test_fft_reconstruction.py`
- [x] T017 [P] [US2] Write unit test for MAPE calculation and SPLY baseline logic in `tests/unit/test_evaluation.py`

### Implementation for User Story 2

- [x] T018 [P] [US2] Implement FFT extraction and Top-K filtering using `scipy.fft` in `src/models/fft_extractor.py`
- [x] T019 [P] [US2] Implement time-series reconstruction from FFT profile in `src/models/fft_reconstructor.py`
- [x] T020 [P] [US2] Implement SPLY baseline generator in `src/evaluation/baseline.py`
- [x] T021 [US2] Implement MAPE calculation and evaluation wrapper in `src/evaluation/metrics.py`
- [x] T022 [US2] Integrate FFT extraction and evaluation into a model training pipeline in `src/models/pipeline.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 4 - External Events Integration (Priority: P2)

**Goal**: Ingest historical and future events, calculate residual impacts from FFT baselines, and prepare impact profiles.

### Tests for User Story 4

- [x] T023 [P] [US4] Write unit test for event calendar parsing and mapping in `tests/unit/test_events_parser.py`
- [x] T024 [P] [US4] Write unit test for calculating residuals (Actual - FFT Reconstruction) in `tests/unit/test_residuals.py`
- [x] T025 [P] [US4] Write unit test for event impact profiling in `tests/unit/test_event_impact.py`

### Implementation for User Story 4

- [x] T026 [P] [US4] Implement Event Calendar CSV reader in `src/data/events_ingestion.py`
- [x] T027 [US4] Implement Residual Calculator comparing historical data to FFT reconstruction in `src/models/residuals.py`
- [x] T028 [US4] Implement Event Impact Profiler (averaging residuals by event category) in `src/models/event_impact.py`
- [x] T028a [US4] Implement Schulferien Index generator (population-weighted) in `src/data/holidays.py`
- [x] T028b [US4] Implement Open-Meteo API client for extreme weather adjustment multipliers in `src/data/weather.py`

**Checkpoint**: Event processing is ready to be injected into the final forecaster.

---

## Phase 6: User Story 3 - Full-Year Forecasting Generation (Priority: P2)

**Goal**: Extrapolate audience metrics for an entire upcoming year (8,760 hours), apply external event impacts, and export to CSV.

### Tests for User Story 3

- [x] T029 [P] [US3] Write unit test for 1-year timestamp generation and FFT extrapolation in `tests/unit/test_extrapolation.py`
- [x] T030 [P] [US3] Write unit test for applying event impact profiles to the extrapolated forecast in `tests/unit/test_apply_impacts.py`
- [x] T031 [P] [US3] Write contract test for CSV output formatting in `tests/contract/test_export.py`

### Implementation for User Story 3

- [x] T032 [P] [US3] Implement future timestamp generation and base FFT extrapolation logic in `src/models/forecaster.py`
- [x] T033 [US3] Implement logic to apply `EventImpactProfile` adjustments to the base forecast in `src/models/forecaster.py`
- [x] T034 [P] [US3] Implement CSV export functionality matching Output Schema in `src/data/export.py`
- [x] T035 [US3] Implement CLI tool using `argparse` or `click` supporting `--events` argument in `src/cli/main.py`
- [x] T036 [US3] Tie ingestion, FFT, event processing, forecasting, and export together into the CLI command in `src/cli/main.py`

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T037 Update main `README.md` with CLI usage instructions and output schema
- [x] T038 [P] Ensure memory efficiency in Pandas operations (verify `float32` usage across pipeline)
- [x] T039 [P] Verify `random.seed` and `numpy.random.seed` are set globally for reproducibility
- [x] T040 Research and compare the FFT approach against Prophet, ConvLSTM, and other modern forecasting methods

---

## Implementation Strategy

### Incremental Delivery Flow

1. **Foundation & Setup** (Phases 1-2): Ready the environment.
2. **US1 (Data processing)**: Ensure data is clean and interpolated.
3. **US2 (Base Math)**: Prove FFT works and establishes a baseline error rate vs SPLY.
4. **US4 (Event Impacts)**: Model the residuals left behind by FFT to understand holiday/promo impacts.
5. **US3 (Final Forecast & CLI)**: Generate the future timeline, overlay the FFT wave, apply the event spikes, and save to CSV.
