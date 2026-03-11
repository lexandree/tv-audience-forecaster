import numpy as np
import scipy.fft
from src.models.types import FFTModelProfile

def extract_top_k_frequencies(signal: np.ndarray, k: int = 100) -> FFTModelProfile:
    """
    Applies Fast Fourier Transform to a time-series signal and extracts
    the top K frequencies (by magnitude), their amplitudes, and phases.
    The DC component (0 Hz) is extracted separately as the mean.
    """
    n = len(signal)
    
    # 1. Apply FFT
    fft_result = scipy.fft.fft(signal)
    
    # 2. Get frequencies
    # For hourly data, the sampling rate is 1. Frequencies are cycles per hour.
    freqs = scipy.fft.fftfreq(n)
    
    # 3. Get amplitudes and phases
    # Normalize amplitudes by N
    amplitudes = np.abs(fft_result) / n
    phases = np.angle(fft_result)
    
    # 4. Separate DC Component (index 0)
    # The DC component amplitude represents the mean of the signal
    dc_mean = amplitudes[0]
    
    # 5. We only care about positive frequencies (excluding 0)
    # The negative frequencies are symmetric for real signals
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    amps_pos = amplitudes[pos_mask]
    phases_pos = phases[pos_mask]
    
    # 6. Because we drop negative frequencies, we must double the amplitudes 
    # of positive frequencies to preserve the energy when reconstructing
    amps_pos = amps_pos * 2.0
    
    # 7. Find top K indices
    # We sort by amplitude in descending order
    top_indices = np.argsort(amps_pos)[::-1][:k]
    
    # 8. Extract Top K
    top_freqs = freqs_pos[top_indices]
    top_amps = amps_pos[top_indices]
    top_phases = phases_pos[top_indices]
    
    return FFTModelProfile(
        frequencies=top_freqs,
        amplitudes=top_amps,
        phases=top_phases,
        mean_value=float(dc_mean)
    )
