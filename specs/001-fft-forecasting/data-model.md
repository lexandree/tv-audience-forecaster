# Data Model: TV Audience Forecasting

## 1. Audience Observation (Input Data Row)
Represents a single raw data point ingested from the source CSV.
- `timestamp` (Datetime): The exact date and hour of the observation.
- `age_group` (String): The age bracket (e.g., "18-34").
- `gender` (String): The gender bracket (e.g., "M", "F").
- `tvr` (Float32): Television Rating percentage.

## 2. Demographic Time-Series (Processed Data)
A continuous, gap-free sequence representing a specific demographic.
- `index`: DatetimeIndex with strict hourly frequency (`freq='H'`).
- `tvr`: Array of Float32 values. Missing values are filled via Seasonal Interpolation.
- `demographic_key`: Tuple of `(age_group, gender)`.

## 3. FFT Model Profile
The mathematical representation of the extracted seasonalities.
- `frequencies` (Array of Floats): The Top-K frequency bins.
- `amplitudes` (Array of Floats): The magnitude of each Top-K frequency.
- `phases` (Array of Floats): The phase shift of each Top-K frequency.
- `mean_value` (Float): The DC component (0 Hz frequency) representing the overall average TVR.

## 4. External Event Calendar
A mapping of specific dates/hours to known events.
- `timestamp` (Datetime): When the event occurs.
- `event_category` (String): The type of event (e.g., "New_Year", "Football_Final").
- `is_historical` (Boolean): True if it happened in the past, False if planned for the future.

## 5. Event Impact Profile
The isolated effect of an event category on a specific demographic.
- `demographic_key`: Tuple of `(age_group, gender)`.
- `event_category` (String): The type of event.
- `average_residual_tvr` (Float32): The average +/- shift in TVR caused by this event, calculated after removing FFT seasonality.

## 6. Yearly Forecast (Output Data Row)
The final generated predictions.
- `timestamp` (Datetime): Future date and hour.
- `age_group` (String): Target age bracket.
- `gender` (String): Target gender bracket.
- `predicted_tvr` (Float32): Extrapolated TVR value (FFT baseline + Event Impact).

## 7. External Factors (Weather & Holidays)
Additional regressors or adjustment multipliers applied to the baseline forecast.
- `schulferien_index` (Float): A population-weighted index (0.0 to 1.0) representing the proportion of the German population currently on school holidays, aggregated across all 16 federal states.
- `weather_adjustment_multiplier` (Float): A multiplier (e.g., 1.09 for extreme heat/rain, 1.06 for extreme cold) derived from DWD ICON weather forecasts (temperature and precipitation) via Open-Meteo API.
