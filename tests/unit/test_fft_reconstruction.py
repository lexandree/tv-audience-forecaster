import pytest
import numpy as np
from src.models.fft_extractor import extract_top_k_frequencies
from src.models.fft_reconstructor import reconstruct_from_fft

def test_fft_reconstruction():
    t = np.arange(0, 336)
    original_signal = 10.0 + 5.0 * np.cos(2 * np.pi * t / 24)
    
    # Extract
    profile = extract_top_k_frequencies(original_signal, k=1)
    
    # Reconstruct for exactly the same length
    reconstructed_signal = reconstruct_from_fft(profile, length=336)
    
    # The reconstructed signal should closely match the original
    np.testing.assert_allclose(reconstructed_signal, original_signal, atol=0.5)

def test_fft_extrapolation():
    t = np.arange(0, 336)
    original_signal = 10.0 + 5.0 * np.cos(2 * np.pi * t / 24)
    
    profile = extract_top_k_frequencies(original_signal, k=1)
    
    # Reconstruct for a longer period (extrapolation)
    reconstructed_signal = reconstruct_from_fft(profile, length=500)
    
    assert len(reconstructed_signal) == 500
    
    # Check a point in the extrapolated future (e.g., exactly 1 week later, should be identical to hour 0)
    # Hour 336 is exactly 14 days later. Since daily seasonality has period 24, 336 % 24 == 0.
    assert np.isclose(reconstructed_signal[336], original_signal[0], atol=0.5)
