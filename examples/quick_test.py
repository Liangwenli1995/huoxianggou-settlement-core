#!/usr/bin/env python3
"""Deterministic synthetic test for the compact public workflow."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core_workflow import run  # noqa: E402


rng = np.random.default_rng(42)
daily_dates = pd.date_range("2017-10-01", "2026-10-01", freq="D")
season = np.clip(np.sin(2 * np.pi * (daily_dates.dayofyear - 165) / 365.25), 0, None)
rainfall = (rng.random(len(daily_dates)) < 0.10 + 0.25 * season) * rng.gamma(
    1.7, 8.0, len(daily_dates)
)
rain = pd.DataFrame({"date": daily_dates, "precipitation_mm": rainfall})
daily = rain.set_index("date")["precipitation_mm"]

dates = pd.date_range("2018-04-17", periods=201, freq="12D")
memory = np.array(
    [daily.loc[date - pd.Timedelta(132, "D"):date - pd.Timedelta(73, "D")].sum() for date in dates]
)
years = np.arange(len(dates)) * 12 / 365.25
increment = -3.2 * np.exp(-years / 3.0) - 0.012 * (memory - memory.mean())
increment += rng.normal(0, 0.20, len(dates))
epochs = pd.DataFrame(
    {
        "date": dates,
        "deformation_mm": np.cumsum(increment),
        "soil_moisture": 0.20 + memory * 0.0002,
        "temperature_c": 13 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25),
    }
)

point_rows = []
for point_id, ultimate in (("P01", 300.0), ("P02", 500.0), ("P03", 700.0)):
    time = (dates - dates[0]).days.to_numpy()
    deformation = -(ultimate * time / (600 + time)) + rng.normal(0, 0.5, len(time))
    point_rows.extend(
        {"point_id": point_id, "date": date, "deformation_mm": value}
        for date, value in zip(dates, deformation)
    )
points = pd.DataFrame(point_rows)

output = ROOT / "outputs" / "quick_test"
run(epochs, rain, output, points)

required = {"metrics.csv", "predictions.csv", "rainfall_scan.csv", "feature_attribution.csv", "consolidation.csv"}
if not required.issubset({path.name for path in output.glob("*.csv")}):
    raise RuntimeError("One or more expected outputs were not created.")
if pd.read_csv(output / "consolidation.csv")["fit_accepted"].sum() < 1:
    raise RuntimeError("The synthetic test produced no accepted consolidation fit.")
print("QUICK TEST PASSED")

