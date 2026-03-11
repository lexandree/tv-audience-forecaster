import numpy as np
from src.models.types import FFTModelProfile

def reconstruct_from_fft(profile: FFTModelProfile, length: int) -> np.ndarray:
    """
    Reconstructs or extrapolates a time-series from an FFT profile using
    cosine waves.
    
    The equation for each frequency component is:
    signal(t) = amplitude * cos(2 * pi * f * t + phase)
    """
    t = np.arange(length)
    
    # Start with the DC component (the baseline mean)
    reconstructed = np.full(length, profile.mean_value)
    
    # Add each frequency wave
    for i in range(len(profile.frequencies)):
        f = profile.frequencies[i]
        amp = profile.amplitudes[i]
        phase = profile.phases[i]
        
        wave = amp * np.cos(2 * np.pi * f * t + phase)
        reconstructed += wave
        
    # Ensure we don't return negative ratings (mathematically possible with FFT truncation)
    reconstructed = np.maximum(reconstructed, 0.0)
    
    return reconstructed
