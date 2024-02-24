import argparse
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg



TRAIN_COLUMNS = ['Latest March Ice Reading', 'feb_low_temp', 'feb_high_temp', 'Latest March Date', 'feb_avg_temp']



def main():
    parser = argparse.ArgumentParser(description="Reads a path to a CSV and a YAML file.")
    parser.add_argument("--csv_path", type=str, required=False, default='NenanaIceClassic_1917-2021.csv')
    parser.add_argument("--num_lags", type=int, required=False, default=5)
    
    args = parser.parse_args()

    num_lags = args.num_lags
    
    # Read CSV file
    try:
        df_nenana = pd.read_csv(args.csv_path)
        print("CSV file loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
    
    df_full = df_nenana[72:].reset_index(drop=True)

    prior_melt = []
    for idx, row in df_full.iterrows():
        if idx == 0:
            continue
        else:
            prior_melt.append(df_full.iloc[idx - 1]['Decimal Day of Year'])
    df_full = df_full[1:]
    df_full['prior_melt'] = prior_melt
    df_train = df_full[:22]
    df_val = df_full[22:]

    model = AutoReg(endog=df_train['Decimal Day of Year'], lags=num_lags, exog=df_train[TRAIN_COLUMNS])

    model_fit = model.fit()
    y_preds = model_fit.predict(start=1, end=len(df_val), exog_oos=df_val[TRAIN_COLUMNS])

    mse = mean_squared_error(df_val['Decimal Day of Year'].values[num_lags-1:], y_preds[num_lags-1:].values)
    print(f'Mean Squared Error: {mse:.2f}')
    mean_day_error = np.mean(abs(df_val['Decimal Day of Year'].values[num_lags-1:] - y_preds[num_lags-1:].values))
    print(f'Mean Day Error: {mean_day_error:.2f}')

    plt.plot(df_val['Year'].values[(num_lags-1):], df_val['Decimal Day of Year'].values[num_lags-1:], 'b')
    plt.plot(df_val['Year'].values[(num_lags-1):], y_preds[num_lags-1:].values, 'r')
    plt.legend(['Actual Melt', 'Predicted Melt', ])
    plt.xlabel('Year')
    plt.ylabel('Decimal Day of Year')
    plt.title(f'Mean Squared Error: {mse:.2f}, Mean Day Error: {mean_day_error:.2f}')
    plt.savefig('auto_reg_results.png')

if __name__ == "__main__":
    main()
