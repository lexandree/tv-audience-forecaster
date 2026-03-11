import pytest
import numpy as np
from src.models.convlstm_forecaster import create_sequences

def test_create_sequences():
    data = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    window_size = 3
    
    X, y = create_sequences(data, window_size)
    
    # Total samples should be len(data) - window_size
    assert len(X) == 7
    assert len(y) == 7
    
    # First sequence should be [0, 1, 2], target 3
    np.testing.assert_array_equal(X[0], np.array([0, 1, 2]))
    assert y[0] == 3
    
    # Last sequence should be [6, 7, 8], target 9
    np.testing.assert_array_equal(X[-1], np.array([6, 7, 8]))
    assert y[-1] == 9
