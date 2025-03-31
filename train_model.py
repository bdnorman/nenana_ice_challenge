import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime, timedelta


_PREDICTING_YEAR = 2025


# Run this with 1 or 2 lag
# TRAIN_COLUMNS = [
#     "Latest March Ice Reading",
#     "march_low_temp",
#     "march_high_temp",
#     # "Latest March Date",
#     "feb_avg_temp",
#     'march_avg_temp',
#     "Latest Feb Ice Reading",
#     # "Latest Feb Date",
# ]

# Run this lage of 3 for 3.17 day error
TRAIN_COLUMNS = [
    "Latest March Ice Reading",
    # "march_low_temp",
    # "march_high_temp",
    "Latest March Date",
    # "feb_avg_temp",
    'march_avg_temp',
    "Latest Feb Ice Reading",
    "Latest Feb Date",
    "march_avg_solar_radiation"
]

def convert_decimal_day_to_date(decimal_day):
    # Assuming a non-leap year starting on January 1
    start_date = datetime(year=_PREDICTING_YEAR-1, month=1, day=1)
    
    # Calculate the full date and time from the decimal day
    full_date_time = start_date + timedelta(days=decimal_day - 1)  # Subtract 1 because January 1st is day 1, not day 0
    
    # Format the date and time into month-day hour-minute format
    formatted_date_time = full_date_time.strftime("%B %d, %I:%M %p")
    
    return formatted_date_time


def main():
    parser = argparse.ArgumentParser(
        description="Reads a path to a CSV and a YAML file."
    )
    parser.add_argument(
        "--csv_path", type=str, required=False, default="NenanaIceClassic_1917-2021.csv"
    )
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
            prior_melt.append(df_full.iloc[idx - 1]["Decimal Day of Year"])
    df_full = df_full[1:]
    df_full["prior_melt"] = prior_melt
    df_train = df_full[:22]
    df_val = df_full[22:]

    model = AutoReg(
        endog=df_train["Decimal Day of Year"],
        lags=num_lags,
        exog=df_train[TRAIN_COLUMNS],
    )

    model_fit = model.fit()
    y_preds = model_fit.predict(
        start=1, end=len(df_val), exog_oos=df_val[TRAIN_COLUMNS]
    )

    # -1 to account for 2024
    mse = mean_squared_error(
        df_val["Decimal Day of Year"].values[num_lags - 1 :][:-1],
        y_preds[num_lags - 1 :].values[:-1],
    )
    print(f"Mean Squared Error: {mse:.2f}")
    mean_day_error = np.mean(
        abs(
            df_val["Decimal Day of Year"].values[num_lags - 1 :][:-1]
            - y_preds[num_lags - 1 :].values[:-1]
        )
    )
    print(f"Mean Day Error: {mean_day_error:.2f}")

    plt.figure()
    plt.scatter(df_full["Latest March Ice Reading"].values[:-1], df_full["Decimal Day of Year"].values[:-1], color='b')
    plt.xlabel('March Ice Thickness')
    plt.ylabel('Decimal Day of Year')
    plt.savefig('Ice Reading vs Decimal Day of Year.png')

    plt.figure()
    actual_years = df_val["Year"].values[(num_lags - 1) :]
    actual_values = df_val["Decimal Day of Year"].values[num_lags - 1 :]
    predicted_values = y_preds[num_lags - 1 :].values
    error = abs(actual_values[:-1] - predicted_values[:-1])
    # Calculate the 95% confidence interval
    confidence_interval = 1.96 * np.std(error) / np.sqrt(len(error))
    print(confidence_interval)
    lower_bound = predicted_values - confidence_interval
    upper_bound = predicted_values + confidence_interval
    plt.fill_between(actual_years, lower_bound, upper_bound, color='r', alpha=0.2, label='95% Confidence Interval')
    plt.plot(actual_years, actual_values, 'o-', color='b', label='Actual Melt')
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Decimal Day of Year")
    date_pred_year = convert_decimal_day_to_date(y_preds.values[-1])

    plt.title(f"Mean Squared Error: {mse:.2f}, Mean Day Error: {mean_day_error:.2f}\n2024 prediction: {date_pred_year}")
    for i, txt in enumerate(error):
        plt.annotate(f"{txt:.2f}", (actual_years[i], predicted_values[i]))
    plt.savefig("auto_reg_results_with_confidence_interval.png")


if __name__ == "__main__":
    main()
