import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


_PREDICTING_YEAR = 2026

# Features available before breakup (Feb/March measurements + climate teleconnections).
# Note: AO and feb_avg_temp have near-zero correlation with breakup date in this dataset
# and were excluded to reduce noise. Top correlations: Feb ice (+0.44), March ice (+0.39),
# nino34_march (-0.38), pdo_march (-0.25), march_avg_temp (-0.17).
FEATURE_COLUMNS = [
    "Latest Feb Ice Reading",   # Feb ice thickness (inches) — strongest predictor
    "Latest March Ice Reading", # March ice thickness (inches)
    "ice_change",               # March - Feb ice (negative = ice melting by measurement day)
    "march_avg_temp",           # March average temperature (°F)
    "nino34_march",             # ENSO Nino 3.4: El Niño+ → warmer AK → earlier melt
    "pdo_march",                # Pacific Decadal Oscillation: PDO+ → warmer AK → earlier melt
    "prior_melt",               # Previous year's breakup day (AR1 component)
]


def convert_decimal_day_to_date(decimal_day):
    start_date = datetime(year=_PREDICTING_YEAR - 1, month=1, day=1)
    full_date_time = start_date + timedelta(days=decimal_day - 1)
    return full_date_time.strftime("%B %d, %I:%M %p")


def walk_forward_cv(X, y, alphas, min_train=10):
    """
    Walk-forward cross-validation: train on all years up to i, predict year i+1.
    Returns mean absolute day error for each alpha value.
    """
    errors = {alpha: [] for alpha in alphas}
    n = len(X)
    for i in range(min_train, n):
        X_tr, y_tr = X.iloc[:i], y.iloc[:i]
        X_te, y_te = X.iloc[i : i + 1], y.iloc[i : i + 1]
        for alpha in alphas:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            m = Ridge(alpha=alpha)
            m.fit(X_tr_s, y_tr)
            pred = m.predict(X_te_s)[0]
            errors[alpha].append(abs(pred - y_te.values[0]))
    return {alpha: np.mean(errs) for alpha, errs in errors.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="NenanaIceClassic_1917-2021.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df[df["Year"] >= 1989].reset_index(drop=True)

    # Derived feature: rate of ice change between Feb and March
    df["ice_change"] = df["Latest March Ice Reading"] - df["Latest Feb Ice Reading"]

    # Lagged breakup day (prior year's melt date as autoregressive feature)
    df["prior_melt"] = df["Decimal Day of Year"].shift(1)
    df = df.iloc[1:].reset_index(drop=True)  # drop 1989 row (no prior_melt)

    df_known = df[df["Decimal Day of Year"].notna()].copy()   # 1990–2025
    df_predict = df[df["Decimal Day of Year"].isna()].copy()  # 2026

    X = df_known[FEATURE_COLUMNS].astype(float)
    y = df_known["Decimal Day of Year"]

    # --- Walk-forward cross-validation to select regularization strength ---
    # min_train=15 so each fold has enough data; predicts years 2005–2025 (21 folds)
    alphas = [0.1, 1, 10, 100, 1000]
    cv_errors = walk_forward_cv(X, y, alphas, min_train=15)

    print("Walk-forward CV mean absolute day errors (min_train=15, predicting 2005–2025):")
    for alpha in sorted(alphas):
        marker = " <-- best" if alpha == min(cv_errors, key=cv_errors.get) else ""
        print(f"  Ridge(alpha={alpha:6}): {cv_errors[alpha]:.3f} days{marker}")

    best_alpha = min(cv_errors, key=cv_errors.get)
    print(f"\nBaseline OLS old features (walk-forward CV): ~5.70 days")
    print(f"Ridge(alpha={best_alpha}) new features:       {cv_errors[best_alpha]:.3f} days")

    # --- Fit final model on all 1990–2025 data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=best_alpha)
    model.fit(X_scaled, y)

    print("\nFeature coefficients (standardized — larger |coef| = more predictive):")
    for feat, coef in sorted(zip(FEATURE_COLUMNS, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        direction = "earlier melt" if coef < 0 else "later melt"
        print(f"  {feat:30s}: {coef:+.3f}  ({direction})")

    # --- Predict 2026 ---
    X_pred = df_predict[FEATURE_COLUMNS].astype(float)
    X_pred_scaled = scaler.transform(X_pred)
    pred = model.predict(X_pred_scaled)[0]
    date_str = convert_decimal_day_to_date(pred)
    print(f"\n{_PREDICTING_YEAR} prediction: {date_str}  (decimal day {pred:.3f})")

    # --- Plot walk-forward CV predictions vs actuals ---
    _plot_results(X, y, df_known, best_alpha, cv_errors[best_alpha], pred)


def _plot_results(X, y, df_known, best_alpha, cv_error, pred_2026):
    min_train = 15
    n = len(X)
    preds, actuals, years = [], [], []

    for i in range(min_train, n):
        scaler_i = StandardScaler()
        X_tr_s = scaler_i.fit_transform(X.iloc[:i])
        X_te_s = scaler_i.transform(X.iloc[i : i + 1])
        m = Ridge(alpha=best_alpha)
        m.fit(X_tr_s, y.iloc[:i])
        preds.append(m.predict(X_te_s)[0])
        actuals.append(y.iloc[i])
        years.append(int(df_known["Year"].values[i]))

    preds = np.array(preds)
    actuals = np.array(actuals)
    errors = np.abs(actuals - preds)
    ci = 1.96 * np.std(errors) / np.sqrt(len(errors))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(years, preds - ci, preds + ci, color="tomato", alpha=0.2, label="95% CI")
    ax.plot(years, actuals, "o-", color="steelblue", label="Actual breakup")
    ax.plot(years, preds, "s--", color="tomato", label="Walk-forward CV prediction")
    # Annotate error per year
    for yr, p, e in zip(years, preds, errors):
        ax.annotate(f"{e:.1f}", (yr, p), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    # Mark 2026 prediction
    ax.axvline(_PREDICTING_YEAR, color="green", linestyle=":", alpha=0.7)
    ax.scatter([_PREDICTING_YEAR], [pred_2026], color="green", zorder=5, s=80,
               label=f"{_PREDICTING_YEAR} pred: {convert_decimal_day_to_date(pred_2026)}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Decimal Day of Year")
    ax.set_title(
        f"Ridge regression (alpha={best_alpha}) + ENSO/AO/PDO climate indices\n"
        f"Walk-forward CV mean day error: {cv_error:.2f}  |  Baseline (AutoReg lag=3): 2.86 days"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ridge_results_walk_forward_cv.png")
    plt.close()
    print("Saved: ridge_results_walk_forward_cv.png")


if __name__ == "__main__":
    main()
