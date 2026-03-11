class ForecastingError(Exception):
    """Base exception for the TV Audience Forecasting project."""
    pass

class DataIngestionError(ForecastingError):
    """Raised when there is an issue with ingesting data."""
    pass

class ValidationError(ForecastingError):
    """Raised when data fails to match expected contract/schema."""
    pass

class MissingDataError(ForecastingError):
    """Raised when there is too much missing data to interpolate."""
    pass

class FFTMathError(ForecastingError):
    """Raised when the FFT model cannot process the time-series."""
    pass
