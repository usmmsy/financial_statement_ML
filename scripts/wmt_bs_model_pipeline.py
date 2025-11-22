"""Run the full WMT balance-sheet modeling pipeline in one shot.

This wraps the three main stages:

1. scripts/estimate_wmt.py  - learn initial policies and driver params.
2. scripts/train_wmt.py     - calibrate selected policies on recent quarters.
3. scripts/forecast_wmt.py  - generate a forward balance-sheet forecast.

Intended usage (from repo root):

    python scripts/wmt_bs_model_pipeline.py

The script imports and calls the module-level ``main`` functions rather
than shelling out to subprocesses, so it runs inside the current Python
process and environment.
"""
from __future__ import annotations

import pathlib
import sys


# Ensure the project root is on sys.path so that ``scripts`` imports resolve
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    # Import lazily inside main so side effects (like TensorFlow import) only
    # happen when the pipeline is actually invoked.
    import scripts.estimate_wmt as estimate_wmt
    import scripts.train_wmt as train_wmt
    import scripts.forecast_wmt as forecast_wmt

    print("[WMT pipeline] Stage 1/3: estimate_wmt.main()")
    estimate_wmt.main()

    print("[WMT pipeline] Stage 2/3: train_wmt.main()")
    train_wmt.main()

    print("[WMT pipeline] Stage 3/3: forecast_wmt.main()")
    forecast_wmt.main()


if __name__ == "__main__":
    main()
