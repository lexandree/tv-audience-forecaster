import pandas as pd
import matplotlib.pyplot as plt

def run():
    forecast_df = pd.read_csv('data/output/forecast_2026.csv')
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    
    # Plot for two demographics as an example
    f_df_1 = forecast_df[(forecast_df['age_group'] == '18-24') & (forecast_df['gender'] == 'F')].copy()
    f_df_2 = forecast_df[(forecast_df['age_group'] == '55+') & (forecast_df['gender'] == 'M')].copy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(f_df_1['timestamp'], f_df_1['predicted_tvr'], label='18-24 F', color='#1f77b4', linewidth=1.5)
    plt.plot(f_df_2['timestamp'], f_df_2['predicted_tvr'], label='55+ M', color='#ff7f0e', linewidth=1.5)
    
    plt.title('TV Audience Forecast 2026 (Selected Demographics)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Predicted TVR (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    import os
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/forecast_plot.png", dpi=150)
    print("Saved docs/forecast_plot.png")

if __name__ == '__main__':
    run()
