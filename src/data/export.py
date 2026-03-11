import pandas as pd

def export_forecast(forecast_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the final forecasted dataframe to a CSV file following the schema contract.
    """
    # Ensure columns are in the correct order
    expected_columns = ['timestamp', 'age_group', 'gender', 'predicted_tvr']
    
    df_to_save = forecast_df.copy()
    
    # If timestamp is the index, reset it and name the column 'timestamp'
    if 'timestamp' not in df_to_save.columns:
        df_to_save.index.name = 'timestamp'
        df_to_save = df_to_save.reset_index()
        
    df_to_save = df_to_save[expected_columns]
    
    # Round to 4 decimal places for cleaner output
    df_to_save['predicted_tvr'] = df_to_save['predicted_tvr'].round(4)
    
    df_to_save.to_csv(output_path, index=False)
