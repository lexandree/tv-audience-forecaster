# Feature Specification: TV Audience Forecasting by Age/Gender

**Feature Branch**: `001-fft-forecasting`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "начинаем проект TV audience forecasting by age/gender. когда то давно я делал подобное на БПФ, надо реконструировать. Благодаря БПФ прогноз делается на целый год"

## Clarifications

### Session 2026-03-11
- Q: What specific TV audience metric is the primary target for forecasting? → A: TVR (Television Rating) / Rating %
- Q: If the historical input data is HOURLY, what should the granularity of the output forecast be? → A: Hourly (8,760 data points per year)
- Q: Since FFT requires a continuous, gap-free time series, what should be the primary strategy for handling missing values in the historical data? → A: Seasonal / Periodic Interpolation (same hour/day last week)
- Q: How should the system handle the number of frequencies extracted via FFT to avoid overfitting to noise while maintaining a "full reconstruction"? → A: Keep Top-K dominant frequencies (highest amplitude)
- Q: Beyond reconstructing historical data, what naive baseline should the FFT forecast be compared against to prove its added value for future predictions? → A: Same Period Last Year (SPLY)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Historical Data Ingestion and Segmentation (Priority: P1)

As a data analyst, I want to ingest historical TV audience data and segment it by specific age and gender demographics so that the forecasting model receives clean, demographic-specific time-series data.

**Why this priority**: Without clean, segmented historical data, the FFT model cannot be applied to distinct demographic groups.

**Independent Test**: Can be fully tested by providing a sample dataset and verifying the output correctly splits the data into distinct, continuous time-series for each age/gender combination.

**Acceptance Scenarios**:

1. **Given** a raw dataset of historical TV audience metrics, **When** the ingestion process runs, **Then** the data is validated and segmented into distinct demographic profiles (age and gender).
2. **Given** missing or anomalous historical data points, **When** the ingestion process runs, **Then** missing values are handled (e.g., imputed or flagged) to ensure a continuous time-series suitable for FFT.

---

### User Story 2 - FFT Model Reconstruction and Training (Priority: P1)

As a forecaster, I want to apply the Fast Fourier Transform (FFT) algorithm to the segmented historical data so that I can reconstruct the past forecasting methodology and extract seasonal frequencies.

**Why this priority**: The FFT algorithm is the core requirement of this project and must be validated against historical patterns before forecasting the future.

**Independent Test**: Can be fully tested by applying the FFT model to a known historical period and measuring how well the extracted frequencies can reconstruct that same period.

**Acceptance Scenarios**:

1. **Given** a clean time-series for a specific demographic, **When** the FFT model is applied, **Then** the model successfully extracts the Top-K dominant frequencies and seasonal patterns.
2. **Given** the extracted frequencies, **When** reconstructing the historical time-series, **Then** the reconstructed series closely matches the original historical data within an acceptable error margin.

---

### User Story 3 - Full-Year Forecasting Generation (Priority: P2)

As a forecaster, I want to use the trained FFT model to extrapolate audience metrics for an entire upcoming year, segmented by age and gender, so that I can plan long-term strategies.

**Why this priority**: This delivers the final business value—the 1-year forecast. It relies completely on the successful completion of the first two stories.

**Independent Test**: Can be fully tested by generating a 1-year forecast and verifying the output covers 365 days with distinct predictions for each demographic group.

**Acceptance Scenarios**:

1. **Given** a trained FFT model for a specific demographic, **When** the forecast generation is triggered, **Then** the system outputs a continuous forecast covering exactly one full year at an hourly resolution.
2. **Given** the generated forecast, **When** the output is exported, **Then** the format matches the required reporting standard.

### Edge Cases

- What happens when a specific demographic has highly volatile or sparse historical data that breaks FFT frequency extraction?
- How does the system handle leap years when generating a "full year" forecast?
- What happens if the historical data contains structural breaks (e.g., a sudden, permanent shift in audience measurement methodology)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest and parse historical TV audience time-series data, specifically focusing on TVR (Television Rating) as the primary metric.
- **FR-002**: System MUST segment the ingested data by age and gender brackets. Bracket definitions follow Standard TV Demographics (M/F: 4-17, 18-34, 35-54, 55+) as a baseline, subject to adjustment based on dataset availability.
- **FR-003**: System MUST require historical data at a HOURLY granularity.
- **FR-004**: System MUST apply Fast Fourier Transform (FFT) to extract the Top-K dominant frequencies (by amplitude) to capture seasonal patterns while filtering noise.
- **FR-005**: System MUST extrapolate the FFT frequencies to generate a forecast extending exactly one year into the future at an HOURLY resolution.
- **FR-006**: System MUST output the final forecasts in CSV / Excel files.
- **FR-007**: System MUST optionally ingest a calendar of historical and future external events.
- **FR-008**: System MUST isolate event impacts from the base FFT seasonality (e.g., via residuals) and apply these impacts to the final generated forecast.

### Key Entities

- **Audience Observation**: A historical data point representing TVR (Television Rating) for a specific timestamp, age group, and gender.
- **Demographic Time-Series**: A continuous sequence of Audience Observations filtered for a single age and gender combination.
- **FFT Model Profile**: The set of extracted dominant frequencies (Top-K), amplitudes, and phases for a specific Demographic Time-Series.
- **Yearly Forecast**: The extrapolated time-series generated from the FFT Model Profile, covering 8,760 (or 8,784) hourly predictions for one full year.
- **External Event**: A known anomaly (e.g., holiday, major broadcast) with a specific timestamp and category.
- **Event Impact Profile**: The calculated average effect (positive or negative TVR shift) of a specific event category on a specific demographic.

## Assumptions

- We assume the historical data is long enough to extract meaningful yearly seasonalities via FFT (ideally 3+ years of history).
- We assume missing data will be handled via Seasonal / Periodic Interpolation (e.g., using data from the same hour and day of the previous week) before passing it to the FFT algorithm to ensure a continuous time series while preserving periodicity.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system processes historical data, including external events, and generates a 1-year forecast for all configured demographic groups in under 5 minutes.
- **SC-002**: The reconstructed historical forecast matches the actual historical baseline data with an error margin consistent with previous project benchmarks (e.g., MAPE < 15%).
- **SC-003**: The final output correctly contains predictions for a full year at hourly resolution for every configured demographic group without any missing gaps.
- **SC-004**: The FFT-based forecast MUST outperform a naive Same Period Last Year (SPLY) baseline in terms of MAPE (Mean Absolute Percentage Error) over a 12-month test period.
- **SC-005**: Forecasts for days with known external events show a statistically significant reduction in error compared to a purely FFT-driven baseline without event adjustments.
