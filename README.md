# Walmart One-Quarter Balance Sheet Forecaster

This project implements a structural, machine-learning–assisted model to forecast Walmart's balance sheet (and linked income and cash-flow statements) one quarter ahead.

The core flow is an end-to-end pipeline:

1. **Estimation (`scripts/estimate_wmt.py`)**
   - Loads historical Walmart quarterly CSVs under `data/retail_csv/WMT_quarterly`.
   - Learns baseline policy schedules and simple driver parameters (e.g. AR(1) sales dynamics, capex scaling, AOCI drift).
   - Saves forward (uncalibrated) policy tensors to `data/models/wmt_policies_forward.npz` and driver parameters to `data/models/wmt_driver_params.npz`.

2. **Calibration (`scripts/train_wmt.py`)**
   - Promotes a subset of policy schedules (e.g. gross margin, opex ratio, depreciation rate, working-capital days, tax and payout) to trainable variables.
   - Reconstructs historical quarters through the structural layer and minimizes a composite loss (accounting identity gap + balance-sheet and income-statement fit).
   - Writes calibrated policies to `data/models/wmt_policies_calibrated.npz`.

3. **Forecast (`scripts/forecast_wmt.py`)**
   - Loads calibrated policies if present (otherwise falls back to forward policies) and driver parameters.
   - Builds the previous state from the most recent fully-populated quarter in the vendor CSVs.
   - Simulates forward drivers (sales via AR(1), depreciation-anchored capex, AOCI drift, aggregate investment, deferred tax and other non-cash items, accrued expenses, dividends, minority interest, etc.).
   - Runs the structural accounting layer to produce forecast `StatementsWMT`.
   - Prints key forecast lines and accounting-identity diagnostics.
   - Exports the forecast to `data/models/wmt_forecast_quarterly.csv`, including statement lines and the relative accounting identity RMSE per run.

4. **Pipeline wrapper (`scripts/wmt_bs_model_pipeline.py`)**
   - Convenience script that runs: **estimate → train → forecast** in a single call.
   - Intended usage from the repo root:

     ```powershell
     python scripts\wmt_bs_model_pipeline.py
     ```

## Core model code

The structural model and typed containers live under `src/wmt_bs_forecaster`:

- `types_wmt.py` — dataclasses-like typed containers for policies, drivers, previous state, and forecast statements.
- `accounting_wmt.py` — deterministic structural layer implementing a Vélez‑Pareja–style roll-forward of the three statements, with explicit working capital, PPE, leases, equity, and identity checks.

At this point, the following files' functionalities are wrapped inside pipeline scripts (`estimate_wmt.py`, `train_wmt.py` and `forecast_wmt.py`), but in the future will be fullfilled as we expand our application.
- `drivers_wmt.py`, `data_wmt.py`, `model_wmt.py`, `forecast_heads.py` — supporting helpers for data loading, driver construction, and potential model extensions.
- `losses_wmt.py` — WMT-specific loss functions used during calibration.

Unit tests live under `tests/unit/wmt`, and integration tests for the end-to-end pipeline are under `tests/integration/test_wmt_pipeline.py`.

## Obsolete / archive code

This repository originally contained a more generic or earlier-iteration set of scripts and helper modules. The **authoritative** path for the current Walmart one-quarter forecast is:

- `src/wmt_bs_forecaster/*`
- `scripts/estimate_wmt.py`
- `scripts/train_wmt.py`
- `scripts/forecast_wmt.py`
- `scripts/wmt_bs_model_pipeline.py`

Any older or "archive" notebooks, scripts, or model definitions outside these paths should be considered **obsolete** and are kept only for reference. When in doubt, prefer the WMT-specific modules listed above and the pipeline wrapper.

## Running tests

To run the focused tests for this WMT forecaster:

```powershell
pytest tests\unit\wmt
pytest tests\integration\test_wmt_pipeline.py
```

This will exercise both the structural components and the full estimate→train→forecast pipeline, including CSV export and basic identity checks.
