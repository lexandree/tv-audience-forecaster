import numpy as np

def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Filters out cases where actual is 0 to avoid division by zero.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    
    # Find indices where actual is not zero
    non_zero_mask = actual != 0
    
    if not np.any(non_zero_mask):
        return 0.0 # Or nan, depending on convention. If everything is 0, error is undefined/0.
        
    filtered_actual = actual[non_zero_mask]
    filtered_predicted = predicted[non_zero_mask]
    
    errors = np.abs((filtered_actual - filtered_predicted) / filtered_actual)
    return float(np.mean(errors))
