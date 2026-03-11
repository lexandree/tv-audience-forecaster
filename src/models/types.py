from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
import numpy as np

@dataclass
class AudienceObservation:
    timestamp: datetime
    age_group: str
    gender: str
    tvr: float

@dataclass
class DemographicTimeSeries:
    demographic_key: Tuple[str, str] # (age_group, gender)
    index: list[datetime]
    tvr: np.ndarray # float32 array

@dataclass
class FFTModelProfile:
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    mean_value: float

@dataclass
class ExternalEvent:
    timestamp: datetime
    event_category: str
    is_historical: bool

@dataclass
class EventImpactProfile:
    demographic_key: Tuple[str, str]
    event_category: str
    average_residual_tvr: float

@dataclass
class YearlyForecast:
    timestamp: datetime
    age_group: str
    gender: str
    predicted_tvr: float
