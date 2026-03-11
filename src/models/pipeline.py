import pandas as pd
from typing import Dict, Tuple

from src.models.types import FFTModelProfile
from src.models.fft_extractor import extract_top_k_frequencies
from src.models.fft_reconstructor import reconstruct_from_fft
from src.evaluation.baseline import generate_sply_baseline
from src.evaluation.metrics import calculate_mape

def train_fft_models(historical_data: Dict[Tuple[str, str], pd.DataFrame], k: int = 100) -> Dict[Tuple[str, str], FFTModelProfile]:
    """
    Takes the processed historical demographic segments, extracts the Top-K 
    FFT frequencies for each, and returns a dictionary of their mathematical profiles.
    """
    profiles = {}
    for key, df in historical_data.items():
        signal = df['tvr'].values
        profile = extract_top_k_frequencies(signal, k=k)
        profiles[key] = profile
        
    return profiles

def evaluate_models(historical_data: Dict[Tuple[str, str], pd.DataFrame], profiles: Dict[Tuple[str, str], FFTModelProfile]) -> Dict[Tuple[str, str], float]:
    """
    Evaluates the FFT reconstruction against the original historical data using MAPE.
    Returns a dictionary of MAPE scores per demographic.
    """
    scores = {}
    for key, df in historical_data.items():
        actual = df['tvr'].values
        profile = profiles[key]
        
        predicted = reconstruct_from_fft(profile, length=len(actual))
        
        mape = calculate_mape(actual, predicted)
        scores[key] = mape
        
    return scores
