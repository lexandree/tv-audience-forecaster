import pytest
import numpy as np
from src.models.fft_extractor import extract_top_k_frequencies

def test_extract_top_k_frequencies():
    # Create a synthetic signal with exactly 2 known frequencies (e.g., daily and weekly)
    t = np.arange(0, 336) # 2 weeks of hours
    # f1 = 1/24 (daily), f2 = 1/168 (weekly)
    base = 10.0
    signal = base + 5.0 * np.cos(2 * np.pi * t / 24) + 2.0 * np.cos(2 * np.pi * t / 168)
    
    # Extract top 2 frequencies (excluding the DC component)
    profile = extract_top_k_frequencies(signal, k=2)
    
    # DC Component should be roughly the base
    assert np.isclose(profile.mean_value, 10.0, atol=0.1)
    
    # We should have exactly 2 frequencies extracted
    assert len(profile.frequencies) == 2
    assert len(profile.amplitudes) == 2
    
    # Convert frequencies back to periods (1/f) to check if we found 24 and 168
    # Note: FFT frequencies are normalized. Real freq = bin_index / N
    periods = 1.0 / profile.frequencies
    
    # We expect periods around 24 and 168
    periods_sorted = np.sort(periods)
    assert np.isclose(periods_sorted[0], 24.0, atol=0.5)
    assert np.isclose(periods_sorted[1], 168.0, atol=0.5)
    
    # We expect amplitudes roughly 5.0 and 2.0
    amps_sorted = np.sort(profile.amplitudes)
    assert np.isclose(amps_sorted[0], 2.0, atol=0.5)
    assert np.isclose(amps_sorted[1], 5.0, atol=0.5)
