from __future__ import annotations

"""
Baseline forecast heads for WMT: deterministic core + light statistical heads.

This module keeps the accounting engine deterministic and only forecasts a few
exogenous or policy-like series with tiny, robust models suitable for very
short histories (e.g., ~5 quarters):

- Sales growth: average of last K growth rates (default K=2)
- Opex ratio: EWMA smoothing
- AOCI drift: ridge regression on a small feature set (e.g., net income)
- Treasury stock drift: ridge on repurchase cash outflow

No external dependencies beyond numpy/tensorflow.

All functions operate on 1D numpy arrays of shape [T]. Caller is responsible
for providing consistent historical slices. These helpers intentionally avoid
altering the deterministic StructuralLayer; instead, they produce next-step
forecasts you can inject into DriversWMT for forward simulation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import numpy as np


def _nan_to_num(a: np.ndarray, val: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.where(np.isfinite(a), a, val)


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Closed-form ridge: beta = (X^T X + l2 I)^(-1) X^T y.

    Args:
        X: [T, F] design matrix.
        y: [T] target vector.
        l2: non-negative L2 penalty.
    Returns:
        beta: [F] coefficients.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    T, F = X.shape
    I = np.eye(F, dtype=np.float32)
    A = X.T @ X + l2 * I
    b = X.T @ y
    # Stable solve (fall back to lstsq if singular)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta


def avg_growth_next(series: np.ndarray, k: int = 2) -> Tuple[float, float]:
    """Compute next-step using average of last k growth rates.

    Returns (growth, next_value) where growth is in decimal (e.g., 0.02 for 2%).
    """
    x = _nan_to_num(series)
    if x.size < 2:
        return 0.0, float(x[-1] if x.size > 0 else 0.0)
    diffs = np.diff(x)
    base = x[:-1]
    # Avoid division by zero
    growth = np.divide(diffs, np.where(np.abs(base) < 1e-6, 1.0, base))
    g = float(np.nanmean(growth[-k:])) if growth.size > 0 else 0.0
    next_val = float(x[-1] * (1.0 + g))
    return g, next_val


def ewma_next(last_value: float, prev_forecast: float, alpha: float = 0.4) -> float:
    """One-step EWMA update for a ratio or bounded series.

    next = alpha * last_value + (1 - alpha) * prev_forecast
    """
    return float(alpha * last_value + (1.0 - alpha) * prev_forecast)


@dataclass
class BaselineHeadsConfig:
    alpha_opex: float = 0.4   # EWMA smoothing for opex ratio
    l2_aoci: float = 1.0      # ridge penalty for AOCI drift
    l2_treasury: float = 1.0  # ridge penalty for treasury drift
    use_intercept: bool = True


@dataclass
class BaselineHeads:
    """Light forecast heads with tiny, interpretable structure.

    Fit on short histories to learn small drifts, then produce next-step
    forecasts for a few key exogenous lines.
    """
    cfg: BaselineHeadsConfig = field(default_factory=BaselineHeadsConfig)

    # Learned coefficients (set after fit)
    beta_aoci: Optional[np.ndarray] = None       # features -> ΔAOCI
    beta_treasury: Optional[np.ndarray] = None   # features -> ΔTreasuryStock

    def fit(
        self,
        history: Dict[str, np.ndarray],
    ) -> None:
        """Fit ridge models for drifts using available history.

        Expected keys in `history` (where available; missing keys are handled):
        - 'net_income': [T]
        - 'aoci': [T]
        - 'treasury_stock': [T]
        - 'repurchase_cash': [T]  (from cash flow; typically negative for buybacks)
        - 'fx_effect': [T]        (optional FX proxy; e.g., Effect Of Exchange Rate Changes)
        """
        aoci = history.get("aoci")
        ni = history.get("net_income")
        fx = history.get("fx_effect")
        tr = history.get("treasury_stock")
        rep = history.get("repurchase_cash")

        # AOCI drift ~ [1, net_income, fx_effect]
        if aoci is not None and (ni is not None or fx is not None) and len(aoci) >= 2:
            y = np.diff(_nan_to_num(aoci))  # delta AOCI
            feats = []
            names = []
            if self.cfg.use_intercept:
                feats.append(np.ones_like(y))
                names.append("intercept")
            if ni is not None:
                feats.append(_nan_to_num(ni[1:]))
                names.append("net_income")
            if fx is not None:
                feats.append(_nan_to_num(fx[1:]))
                names.append("fx_effect")
            X = np.vstack(feats).T if feats else np.zeros((y.shape[0], 0), dtype=np.float32)
            self.beta_aoci = _ridge_fit(X, y, l2=self.cfg.l2_aoci) if X.shape[1] > 0 else None
        else:
            self.beta_aoci = None

        # Treasury drift ~ [1, repurchase_cash]
        if tr is not None and rep is not None and len(tr) >= 2 and len(rep) >= 2:
            y2 = np.diff(_nan_to_num(tr))  # delta TreasuryStock (often negative when repurchasing)
            feats2 = []
            if self.cfg.use_intercept:
                feats2.append(np.ones_like(y2))
            feats2.append(_nan_to_num(rep[1:]))
            X2 = np.vstack(feats2).T
            self.beta_treasury = _ridge_fit(X2, y2, l2=self.cfg.l2_treasury)
        else:
            self.beta_treasury = None

    def forecast_next(
        self,
        last: Dict[str, float],
        prev_forecasts: Dict[str, float],
        features_next: Dict[str, float],
    ) -> Dict[str, float]:
        """Produce one-step forecasts for key lines.

        Args:
            last: dictionary with latest observed values (e.g., 'sales', 'opex_ratio', 'aoci', 'treasury_stock').
            prev_forecasts: previous forecasts for smoothing (e.g., 'opex_ratio').
            features_next: simple drivers for next step (e.g., 'repurchase_cash_next', 'net_income_next').
        Returns:
            dict with keys among: 'sales_next', 'opex_ratio_next', 'aoci_next', 'treasury_stock_next'.
        """
        out: Dict[str, float] = {}

        # Sales via average recent growth
        sales_hist = np.asarray(features_next.get("sales_hist")) if features_next.get("sales_hist") is not None else None
        if sales_hist is not None and sales_hist.size >= 2:
            _, s_next = avg_growth_next(sales_hist, k=2)
            out["sales_next"] = s_next
        elif "sales" in last:
            out["sales_next"] = float(last["sales"])  # hold

        # Opex ratio via EWMA
        if "opex_ratio" in last:
            prev = prev_forecasts.get("opex_ratio", last["opex_ratio"])  # seed with last if no prev forecast
            out["opex_ratio_next"] = ewma_next(float(last["opex_ratio"]), float(prev), alpha=self.cfg.alpha_opex)

        # AOCI drift
        if "aoci" in last and self.beta_aoci is not None:
            feats = []
            if self.cfg.use_intercept:
                feats.append(1.0)
            if "net_income_next" in features_next:
                feats.append(float(features_next["net_income_next"]))
            if "fx_effect_next" in features_next:
                feats.append(float(features_next["fx_effect_next"]))
            if feats:
                delta = float(np.dot(np.asarray(feats, dtype=np.float32), self.beta_aoci))
                out["aoci_next"] = float(last["aoci"]) + delta

        # Treasury stock drift
        if "treasury_stock" in last and self.beta_treasury is not None:
            feats2 = []
            if self.cfg.use_intercept:
                feats2.append(1.0)
            if "repurchase_cash_next" in features_next:
                feats2.append(float(features_next["repurchase_cash_next"]))
            if feats2:
                delta2 = float(np.dot(np.asarray(feats2, dtype=np.float32), self.beta_treasury))
                out["treasury_stock_next"] = float(last["treasury_stock"]) + delta2

        return out
