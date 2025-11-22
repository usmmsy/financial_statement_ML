import os
import sys
import pathlib

# Ensure the project 'src' directory is on sys.path when running as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .types_wmt import StatementsWMT, DriversWMT, PoliciesWMT, PrevStateWMT
try:
    from .drivers_wmt import ArimaDrivers  # optional: requires statsmodels
except Exception:  # ModuleNotFoundError if statsmodels not installed
    ArimaDrivers = None  # type: ignore
from .model_wmt import WalmartBSModel
from .forecast_heads import BaselineHeads, BaselineHeadsConfig
from .accounting_wmt import StructuralLayer
from .losses_wmt import wmt_fit_loss, identity_gap_loss, retained_earnings_consistency_loss

__all__ = [
    "StatementsWMT",
    "DriversWMT",
    "PoliciesWMT",
    "PrevStateWMT",
    "ArimaDrivers",
    "WalmartBSModel",
    "StructuralLayer",
    "BaselineHeads",
    "BaselineHeadsConfig",
    "wmt_fit_loss",
    "identity_gap_loss",
    "retained_earnings_consistency_loss",
]
