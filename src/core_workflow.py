#!/usr/bin/env python3
"""Compact reference implementation of the core settlement workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


SEED = 42


def smooth_and_difference(values):
    values = np.asarray(values, dtype=float)
    window = min(9, len(values) if len(values) % 2 else len(values) - 1)
    smooth = savgol_filter(values, window, min(2, window - 1)) if window >= 5 else values
    return smooth, np.r_[np.nan, np.diff(smooth)]


def rainfall_sum(daily, dates, window, lag):
    """Sum [date-lag-window, date-lag-1], so no future rainfall is used."""
    daily = daily.asfreq("D", fill_value=0.0)
    cumulative = daily.cumsum()
    dates = pd.DatetimeIndex(dates).normalize()
    high = cumulative.reindex(dates - pd.to_timedelta(lag + 1, unit="D")).to_numpy()
    low = cumulative.reindex(
        dates - pd.to_timedelta(lag + window + 1, unit="D"), fill_value=0.0
    ).to_numpy()
    return high - low


def chronological_split(n):
    train_end = max(3, min(round(0.70 * n), n - 4))
    validation_end = max(train_end + 1, min(round(0.85 * n), n - 1))
    index = np.arange(n)
    return index[:train_end], index[train_end:validation_end], index[validation_end:]


def scores(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    pcc = pearsonr(observed, predicted)[0] if np.std(observed) and np.std(predicted) else np.nan
    return {
        "PCC": pcc,
        "MAE": mean_absolute_error(observed, predicted),
        "RMSE": np.sqrt(mean_squared_error(observed, predicted)),
    }


def prepare_epoch_table(epochs, daily_rain):
    epochs = epochs.copy().sort_values("date").reset_index(drop=True)
    epochs["date"] = pd.to_datetime(epochs["date"])
    smooth, increment = smooth_and_difference(epochs["deformation_mm"])
    epochs["smoothed_mm"] = smooth
    epochs["increment_mm"] = increment
    epochs["days_since_start"] = (epochs["date"] - epochs["date"].iloc[0]).dt.days
    for window in (30, 60, 90, 180):
        for lag in (0, 30, 60, 90, 120):
            epochs[f"rain_{window}d_lag{lag}d"] = rainfall_sum(
                daily_rain, epochs["date"], window, lag
            )
    return epochs


def fit_model(prepared):
    excluded = {"date", "deformation_mm", "smoothed_mm", "increment_mm"}
    features = prepared.drop(columns=[c for c in excluded if c in prepared]).apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.loc[:, features.isna().mean() <= 0.40]
    features = features.loc[:, features.nunique(dropna=True) > 1]
    features = features.fillna(features.median())
    valid = prepared["increment_mm"].notna()
    x = features.loc[valid].reset_index(drop=True)
    y = prepared.loc[valid, "increment_mm"].reset_index(drop=True)
    dates = prepared.loc[valid, "date"].reset_index(drop=True)
    train, validation, test = chronological_split(len(x))

    model = RandomForestRegressor(
        n_estimators=800,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1,
    ).fit(x.iloc[train], y.iloc[train])
    predicted = model.predict(x)

    rows = []
    for name, index in {
        "Train": train,
        "Validation": validation,
        "Test": test,
        "Validation+Test": np.r_[validation, test],
    }.items():
        rows.append({"subset": name, "n": len(index), **scores(y.iloc[index], predicted[index])})

    labels = np.full(len(x), "Train", dtype=object)
    labels[validation] = "Validation"
    labels[test] = "Test"
    predictions = pd.DataFrame(
        {"date": dates, "observed_mm": y, "predicted_mm": predicted, "subset": labels}
    )

    attribution = pd.DataFrame(
        {"feature": x.columns, "rf_importance": model.feature_importances_}
    )
    try:
        import shap

        values = np.asarray(shap.TreeExplainer(model).shap_values(x.iloc[np.r_[validation, test]]))
        if values.ndim == 3:
            values = values[..., 0]
        attribution["mean_absolute_SHAP"] = np.abs(values).mean(axis=0)
    except Exception:
        attribution["mean_absolute_SHAP"] = np.nan
    return model, x, y, pd.DataFrame(rows), predictions, attribution


def rainfall_scan(prepared, daily_rain):
    work = prepared.dropna(subset=["increment_mm"])
    rows = []
    for window in (6, 12, 24, 30, 60, 90, 120, 180):
        for lag in range(0, 181, 12):
            rainfall = rainfall_sum(daily_rain, work["date"], window, lag)
            r = pearsonr(rainfall, work["increment_mm"])[0]
            rows.append({"window_days": window, "lag_days": lag, "PCC": r})
    return pd.DataFrame(rows)


def hyperbolic_fit(group):
    group = group.sort_values("date")
    smooth, _ = smooth_and_difference(group["deformation_mm"])
    settlement = -(smooth - smooth[0])
    time = (pd.to_datetime(group["date"]) - pd.to_datetime(group["date"]).iloc[0]).dt.days.to_numpy()
    keep = (time >= 180) & (settlement > 1.0)
    if keep.sum() < 8:
        return None
    slope, intercept = np.polyfit(time[keep], time[keep] / settlement[keep], 1)
    fitted = intercept + slope * time[keep]
    total = np.sum((time[keep] / settlement[keep] - np.mean(time[keep] / settlement[keep])) ** 2)
    r2 = 1.0 - np.sum((time[keep] / settlement[keep] - fitted) ** 2) / total
    ultimate = 1.0 / slope if slope > 0 and intercept > 0 else np.nan
    observed = settlement[-1]
    accepted = r2 >= 0.80 and 0.95 * observed <= ultimate <= 20.0 * observed
    return {
        "point_id": group["point_id"].iloc[0],
        "observed_settlement_mm": observed,
        "R2": r2,
        "ultimate_settlement_mm": ultimate if accepted else np.nan,
        "degree_of_consolidation": observed / ultimate if accepted else np.nan,
        "remaining_settlement_mm": ultimate - observed if accepted else np.nan,
        "fit_accepted": accepted,
    }


def run(epochs, rain, output, points=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    epochs["date"] = pd.to_datetime(epochs["date"])
    rain["date"] = pd.to_datetime(rain["date"])
    daily = rain.groupby("date")["precipitation_mm"].mean().sort_index()
    prepared = prepare_epoch_table(epochs, daily)
    _, _, _, metrics, predictions, attribution = fit_model(prepared)
    scan = rainfall_scan(prepared, daily)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    attribution.to_csv(output / "feature_attribution.csv", index=False)
    scan.to_csv(output / "rainfall_scan.csv", index=False)
    if points is not None:
        points["date"] = pd.to_datetime(points["date"])
        rows = [hyperbolic_fit(group) for _, group in points.groupby("point_id")]
        pd.DataFrame([row for row in rows if row is not None]).to_csv(
            output / "consolidation.csv", index=False
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--rain", required=True)
    parser.add_argument("--points")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(
        pd.read_csv(args.epochs),
        pd.read_csv(args.rain),
        args.output,
        pd.read_csv(args.points) if args.points else None,
    )


if __name__ == "__main__":
    main()

