# Quickstart: TV Audience Forecasting (FFT)

## Overview
This project processes historical TV audience ratings (TVR) using Fast Fourier Transform (FFT) to extract macro-seasonality and generate hourly forecasts for specific demographic groups.

## Prerequisites
- Python 3.12+
- Dependencies: `pandas`, `numpy`, `scipy`

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   Ensure you have a CSV file with historical hourly TVR data. Minimum required columns: `timestamp`, `age_group`, `gender`, `tvr`.

## Running a Forecast

Execute the CLI tool to generate a forecast:

```bash
python -m tv_audience_forecasting.cli forecast \
    --input data/history.csv \
    --output output/forecast.csv \
    --top-k 100
```

This will:
1. Load and clean the historical data (handling missing hours via seasonal interpolation).
2. Segment the data into isolated demographic time-series.
3. Apply FFT to extract the Top 100 dominant frequencies.
4. Extrapolate those frequencies to generate 8,760 hourly predictions.
5. Save the final dataset to `output/forecast.csv`.
