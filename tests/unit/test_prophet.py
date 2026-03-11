import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the forecaster (we will create this next)
from src.models.prophet_forecaster import train_prophet_model, predict_prophet_model

@patch('src.models.prophet_forecaster.Prophet')
def test_train_prophet_model(mock_prophet_class):
    # Setup mock
    mock_model_instance = MagicMock()
    mock_prophet_class.return_value = mock_model_instance

    # Create dummy demographic timeseries data
    dates = pd.date_range('2023-01-01', periods=10, freq='h')
    df = pd.DataFrame({'tvr': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=dates)

    # Call the wrapper function
    model = train_prophet_model(df)

    # Verify Prophet was instantiated
    mock_prophet_class.assert_called_once()
    
    # Verify fit was called with correctly formatted dataframe ('ds', 'y')
    mock_model_instance.fit.assert_called_once()
    args, kwargs = mock_model_instance.fit.call_args
    fit_df = args[0]
    
    assert 'ds' in fit_df.columns
    assert 'y' in fit_df.columns
    assert len(fit_df) == 10
    assert fit_df['y'].iloc[0] == 1.0

@patch('src.models.prophet_forecaster.Prophet')
def test_predict_prophet_model(mock_prophet_class):
    # Setup mock model
    mock_model_instance = MagicMock()
    
    # Create dummy forecast output that Prophet would return
    future_dates = pd.date_range('2023-01-01', periods=5, freq='h')
    dummy_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    mock_model_instance.predict.return_value = dummy_forecast

    # Call our wrapper predict
    predictions = predict_prophet_model(mock_model_instance, future_dates)

    # Check if predict was called
    mock_model_instance.predict.assert_called_once()
    args, kwargs = mock_model_instance.predict.call_args
    predict_df = args[0]
    
    assert 'ds' in predict_df.columns
    assert len(predict_df) == 5

    # Check our wrapper output format
    assert len(predictions) == 5
    assert predictions[0] == 1.5
