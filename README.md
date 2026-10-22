# Compact settlement-analysis workflow

This repository provides a compact, executable reference implementation for the main author-developed calculations used in the associated manuscript on post-construction settlement of a high loess gully fill.

## Included methods

- Savitzky-Golay smoothing and epoch-wise deformation increments.
- Rainfall accumulation over windows that end before each acquisition.
- Chronological train-validation-test splitting.
- Random-forest regression and independent-period evaluation.
- Optional SHAP attribution.
- Hyperbolic consolidation fitting.
- A deterministic synthetic quick test.

The release excludes private data, local paths, study-area assets, production batch code, cartographic routines, manuscript-specific figure formatting, and internal optimization utilities. Predictor tables prepared for the study are treated as inputs.

## Files

```text
README.md
LICENSE
requirements.txt
src/core_workflow.py
examples/quick_test.py
gee/rainfall_export_example.js
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before installing packages. On Windows PowerShell, use `.venv\Scripts\Activate.ps1`. On Linux or macOS, use `source .venv/bin/activate`.

## Quick test

Run from the repository root:

```bash
python examples/quick_test.py
```

A successful run prints:

```text
QUICK TEST PASSED
```

It also creates CSV results in `outputs/quick_test/`.

## Run with prepared tables

```bash
python src/core_workflow.py \
  --epochs path/to/epoch_predictors.csv \
  --rain path/to/daily_rainfall.csv \
  --points path/to/point_settlement.csv \
  --output outputs/my_run
```

The epoch table requires `date`, `deformation_mm`, and one or more numeric predictor columns. Negative deformation denotes settlement.

The daily rainfall table requires `date` and `precipitation_mm`.

The optional point table requires `point_id`, `date`, and `deformation_mm` in long format.

## Expected outputs

- `metrics.csv`
- `predictions.csv`
- `rainfall_scan.csv`
- `feature_attribution.csv`
- `consolidation.csv` when a point table is supplied

## Google Earth Engine example

`gee/rainfall_export_example.js` is a short example showing how a public daily rainfall input can be exported. Replace the placeholder AOI asset before running it in the Google Earth Engine Code Editor.

## Reproducibility settings

- Random seed: 42.
- Random-forest trees: 800.
- Minimum leaf size: 3.
- Chronological split: 70% training, 15% validation, and 15% testing.
- Smoothing: nine-epoch, second-order Savitzky-Golay filter.
- Hyperbolic fits begin at 180 days and require R-squared of at least 0.80.

## Scope

This compact release exposes the equations and executable logic needed to inspect the principal methodological choices without distributing the authors' complete production system. ENVI/SARscape and other third-party software are not redistributed.

## License

The source code is available under the MIT License. External data and third-party software retain their own terms.

