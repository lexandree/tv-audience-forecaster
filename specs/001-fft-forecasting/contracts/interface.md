# Interfaces: TV Audience Forecasting

## CLI Interface
The primary interface for this feature will be a Command Line Interface (CLI) built using Python's `argparse` or `click`.

### Command: `forecast`
Runs the end-to-end forecasting pipeline.

**Arguments:**
- `--input` (required): Path to the historical CSV data file.
- `--output` (required): Path where the forecast CSV should be saved.
- `--events` (optional): Path to the CSV file containing historical and future event calendars.
- `--top-k` (optional): Number of top frequencies to retain in FFT. Default: 100.
- `--years` (optional): Number of years to forecast. Default: 1.

**Example Usage:**
```bash
python -m tv_audience_forecasting.cli forecast \
    --input ./data/historical_tvr_2023_2025.csv \
    --events ./data/events_calendar.csv \
    --output ./forecasts/prediction_2026.csv \
    --top-k 150
```

## Data Contracts

### Input Schema (CSV)
Expected columns in the `input` CSV:
- `timestamp` (YYYY-MM-DD HH:MM:SS)
- `age_group` (String, e.g., "18-34")
- `gender` (String, e.g., "M", "F")
- `tvr` (Float)

### Events Schema (CSV)
Expected columns in the `events` CSV:
- `timestamp` (YYYY-MM-DD HH:MM:SS)
- `event_category` (String, e.g., "Holiday", "Sports")
- `event_name` (String, e.g., "New Years Eve")

### Output Schema (CSV)
Generated columns in the `output` CSV:
- `timestamp` (YYYY-MM-DD HH:MM:SS)
- `age_group` (String)
- `gender` (String)
- `predicted_tvr` (Float)
- `event_impact_applied` (Float, Optional: shows how much TVR was adjusted due to an event)
