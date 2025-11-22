import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "data" / "models"


def test_wmt_full_pipeline_smoke():
    """End-to-end smoke test for WMT pipeline.

    Runs the wrapper script that calls estimate -> train -> forecast and
    asserts that key artifacts are created without raising exceptions.
    """

    # Import via the scripts package so it resolves under pytest
    import scripts.wmt_bs_model_pipeline as wmt_bs_model_pipeline

    # Run the full pipeline; this should complete without error.
    wmt_bs_model_pipeline.main()

    # Check that core artifacts exist.
    assert (MODEL_DIR / "wmt_policies_forward.npz").exists()
    assert (MODEL_DIR / "wmt_driver_params.npz").exists()
    assert (MODEL_DIR / "wmt_policies_calibrated.npz").exists()
    assert (MODEL_DIR / "wmt_forecast_quarterly.csv").exists()


def test_wmt_forecast_csv_sanity():
    """Sanity-check the saved forecast CSV for structure and identity gap.

    Uses the CSV produced by the forecast script and checks that
    required columns are present, the horizon matches the number of
    rows, and the relative identity RMSE is within a reasonable bound.
    """

    csv_path = MODEL_DIR / "wmt_forecast_quarterly.csv"
    assert csv_path.exists(), "Run of forecast_wmt should have produced forecast CSV."

    df = pd.read_csv(csv_path)

    # Basic structure checks
    required_cols = [
        "period",
        "cash",
        "assets",
        "liab_plus_equity",
        "relative_identity_rmse",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col!r} in forecast CSV."

    # Period count should match number of rows
    assert len(df) == df["period"].iloc[-1]

    # Accounting identity should be approximately satisfied
    assets = df["assets"].to_numpy(dtype=float)
    liab_eq = df["liab_plus_equity"].to_numpy(dtype=float)
    # Allow a modest relative tolerance; absolute scale is large.
    assert np.allclose(assets, liab_eq, rtol=0.05, atol=0.0)

    # Relative identity RMSE should be reasonably small
    rel_rmse = float(df["relative_identity_rmse"].iloc[0])
    assert rel_rmse < 0.05
