"""Forecast WMT balance sheet one or more quarters ahead using learned policies.

This script assumes that scripts/train_wmt.py has already been run and that
wmt_policies_forward.npz and wmt_driver_params.npz exist under data/models.

It demonstrates a simple forecast pipeline:
- Load PoliciesWMT forward tensors and driver parameters.
- Build PrevStateWMT from the last available balance sheet.
- Simulate a sales path and capex using learned gamma.
- Run StructuralLayer to obtain forecast StatementsWMT.
"""
from __future__ import annotations
import pathlib
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

import os
import sys
import pathlib

# Ensure the project 'src' directory is on sys.path when running as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wmt_bs_forecaster.types_wmt import PoliciesWMT, DriversWMT, PrevStateWMT
from wmt_bs_forecaster.accounting_wmt import StructuralLayer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "retail_csv" / "WMT_quarterly"
MODEL_DIR = ROOT / "data" / "models"

BAL_PATH = DATA_DIR / "WMT_quarterly_balance_sheet.csv"
FIN_PATH = DATA_DIR / "WMT_quarterly_financials.csv"
CF_PATH = DATA_DIR / "WMT_quarterly_cash_flow.csv"

# In the raw CSVs, columns are ordered left-to-right as latest-to-oldest.
# We use 2025-07-31 as the validation quarter (target we forecast toward),
# and 2025-04-30 as the latest previous period for prev state and AR drivers.
VALIDATION_COL = "2025-07-31"
PREV_COL = "2025-04-30"

EPS = 1e-8


def _load_policies_from_npz(path: pathlib.Path) -> PoliciesWMT:
    """Helper to construct PoliciesWMT from a saved .npz file."""
    npz = np.load(path)
    def t(name: str) -> tf.Tensor:
        return tf.convert_to_tensor(npz[name], dtype=tf.float32)
    kw = {
        "inflation": t("inflation"),
        "real_st_rate": t("real_st_rate"),
        "real_lt_rate": t("real_lt_rate"),
        "tax_rate": t("tax_rate"),
        "payout_ratio": t("payout_ratio"),
        "min_cash_ratio": t("min_cash_ratio"),
        "lt_share_for_capex": t("lt_share_for_capex"),
        "st_invest_spread": t("st_invest_spread"),
        "debt_spread": t("debt_spread"),
        "dso_days": t("dso_days"),
        "dpo_days": t("dpo_days"),
        "dio_days": t("dio_days"),
        "opex_ratio": t("opex_ratio"),
        "depreciation_rate": t("depreciation_rate"),
        "gross_margin": t("gross_margin"),
    }
    # Optional investing-related policy tensors; support older npz files.
    for name in [
        "premium_ratio_goodwill",
        "beta1_capex",
        "beta2_net_invest",
    ]:
        if name in npz.files:
            kw[name] = t(name)
    # Optional OCA coefficients
    for name in [
        "omega_oca_sales",
        "omega_oca_opex",
        "omega_oca_depr",
    ]:
        if name in npz.files:
            kw[name] = t(name)
    # Optional ONCL coefficients
    for name in [
        "psi_oncl_deferred_tax",
        "psi_oncl_other_nc",
        "gamma_capital_stock",
        "k_pi",
    ]:
        if name in npz.files:
            kw[name] = t(name)
    # Optional lease schedule parameters
    for name in [
        "lease_addition_capex_coeff",
        "lease_addition_sales_coeff",
        "lease_avg_remaining_term",
        "lease_principal_payment_rate",
    ]:
        if name in npz.files:
            kw[name] = t(name)
    return PoliciesWMT(**kw)


def load_policies() -> PoliciesWMT:
    """Load calibrated policies if available, otherwise fall back to forward.

    This aligns the forecast script with the intended pipeline:
    estimate_wmt -> train_wmt -> forecast_wmt.
    """
    calibrated_path = MODEL_DIR / "wmt_policies_calibrated.npz"
    forward_path = MODEL_DIR / "wmt_policies_forward.npz"

    if calibrated_path.exists():
        npz = np.load(calibrated_path)
        if "gross_margin" in npz.files:
            # Basic sanity check that this looks like a PoliciesWMT npz
            return _load_policies_from_npz(calibrated_path)
        # If file exists but doesn't contain expected arrays, fall back.

    return _load_policies_from_npz(forward_path)


def load_driver_params() -> Dict[str, float]:
    npz = np.load(MODEL_DIR / "wmt_driver_params.npz")
    params: Dict[str, float] = {"gamma_capex_growth": float(npz["gamma_capex_growth"])}
    # Optional AOCI drift parameter; older npz files may not contain it.
    if "aoci_drift" in npz.files:
        params["aoci_drift"] = float(npz["aoci_drift"])
    else:
        params["aoci_drift"] = 0.0
    # Optional investing sensitivities; default to zeros if missing.
    for k in ["premium_ratio_goodwill", "beta1_capex", "beta2_net_invest"]:
        if k in npz.files:
            params[k] = float(npz[k])
        else:
            params[k] = 0.0
    return params


def load_last_prev_state() -> PrevStateWMT:
    bal = pd.read_csv(BAL_PATH, index_col=0)
    fin = pd.read_csv(FIN_PATH, index_col=0)
    # Use an explicitly chosen "previous" column instead of the last, because
    # the right-most (latest) column is mostly empty in the raw data and
    # 2025-07-31 is reserved as validation target.
    if PREV_COL not in bal.columns or PREV_COL not in fin.columns:
        raise KeyError(
            f"Previous-period column {PREV_COL!r} not found in both balance sheet and financials columns. "
            f"Available in BAL: {list(bal.columns)}; in FIN: {list(fin.columns)}"
        )
    b = bal[PREV_COL]

    def row_like(token: str) -> float:
        # 1) exact match if possible
        for idx, val in b.items():
            if idx.strip().lower() == token:
                return float(val)
        # 2) fallback: substring match
        for idx, val in b.items():
            if token in idx.lower():
                return float(val)
        raise KeyError(f"Row containing '{token}' not found in balance sheet.")

    cash = row_like("cash and cash equivalents")
    # Short term investments: if combined with cash, we keep st_investments as 0
    st_investments = 0.0
    ar = row_like("accounts receivable")
    inventory = row_like("inventory")
    net_ppe = row_like("net ppe")
    st_debt = row_like("current debt")
    lt_debt = row_like("long term debt")
    ap = row_like("accounts payable")
    equity = row_like("stockholders equity")

    # optional extensions
    paid_in_capital = row_like("additional paid in capital") if any("paid in capital" in i.lower() for i in b.index) else 0.0
    other_current_assets = row_like("other current assets") if any("other current assets" in i.lower() for i in b.index) else 0.0
    goodwill = row_like("goodwill") if any("goodwill" in i.lower() for i in b.index) else 0.0
    other_nca = row_like("other non current assets") if any("other non current assets" in i.lower() for i in b.index) else 0.0
    accrued_expenses = row_like("current accrued expenses") if any("current accrued expenses" in i.lower() for i in b.index) else 0.0
    tax_payable = row_like("total tax payable") if any("total tax payable" in i.lower() for i in b.index) else 0.0
    other_ncl = row_like("other non current liabilities") if any("other non current liabilities" in i.lower() for i in b.index) else 0.0
    cur_lease = row_like("current capital lease obligation") if any("current capital lease obligation" in i.lower() for i in b.index) else 0.0
    lt_lease = row_like("long term capital lease obligation") if any("long term capital lease obligation" in i.lower() for i in b.index) else 0.0
    div_payable = row_like("dividends payable") if any("dividends payable" in i.lower() for i in b.index) else 0.0
    minority = row_like("minority interest") if any(
        idx.strip().lower() == "minority interest" for idx in b.index
    ) else 0.0
    cap_stock = row_like("capital stock") if any("capital stock" in i.lower() for i in b.index) else 0.0

    def scalar(x: float) -> tf.Tensor:
        return tf.convert_to_tensor([[x]], dtype=tf.float32)

    return PrevStateWMT(
        cash=scalar(cash),
        st_investments=scalar(st_investments),
        ar=scalar(ar),
        inventory=scalar(inventory),
        net_ppe=scalar(net_ppe),
        st_debt=scalar(st_debt),
        lt_debt=scalar(lt_debt),
        ap=scalar(ap),
        equity=scalar(equity),
        paid_in_capital=scalar(paid_in_capital),
        other_current_assets=scalar(other_current_assets),
        goodwill_intangibles=scalar(goodwill),
        other_non_current_assets=scalar(other_nca),
        accrued_expenses=scalar(accrued_expenses),
        tax_payable=scalar(tax_payable),
        other_non_current_liabilities=scalar(other_ncl),
        current_capital_lease_obligation=scalar(cur_lease),
        long_term_capital_lease_obligation=scalar(lt_lease),
        dividends_payable=scalar(div_payable),
        minority_interest=scalar(minority),
        capital_stock=scalar(cap_stock),
    )


def simulate_sales_and_capex(
    bal: pd.DataFrame,
    fin: pd.DataFrame,
    policies: PoliciesWMT,
    driver_params: Dict[str, float],
    horizon: int,
) -> DriversWMT:
    """Build forecast drivers from historical WMT financials using AR(1).

    - Sales: estimate an AR(1) on the historical Total Revenue series and
      simulate ``horizon`` steps ahead. This provides a nontrivial dsales
      path right from the first forecast quarter.
    - Capex: use forward depreciation_rate multiplied by a base PPE stock
      loaded from the 2024-07-31 balance sheet, plus an optional gamma * dsales
      term using the same gamma_capex_growth parameter learned in estimation.
    """
    # Locate Total Revenue row
    rev_row = None
    for r in fin.index:
        if "total revenue" in r.lower():
            rev_row = r
            break
    if rev_row is None:
        raise ValueError("Total Revenue row not found in financials.")

    sales_series = fin.loc[rev_row].to_numpy(dtype=np.float32)
    # Remove NaNs
    sales_hist = sales_series[~np.isnan(sales_series)]

    if sales_hist.size == 0:
        # Degenerate case: no history; flat zero
        sales_fut = np.zeros((1, horizon, 1), dtype=np.float32)
    elif sales_hist.size == 1:
        # Single point: flat forecast at that level
        base = float(sales_hist[-1])
        sales_fut = np.full((1, horizon, 1), base, dtype=np.float32)
    else:
        # Use the helper AR(1) estimator defined below to get (mu, phi)
        mu, phi = _estimate_ar1_from_series(sales_hist)
        # Start from the last observed revenue
        prev = float(sales_hist[-1])
        path: list[float] = []
        for _ in range(horizon):
            nxt = mu + phi * (prev - mu)
            path.append(float(nxt))
            prev = nxt
        sales_fut = np.array(path, dtype=np.float32).reshape(1, horizon, 1)

    sales_tf = tf.convert_to_tensor(sales_fut)

    # Derive COGS from gross margin in StructuralLayer, so we can set cogs dummy here
    cogs_tf = tf.zeros_like(sales_tf)

    # Depreciation path from policies, scaled by a base PPE stock taken from
    # the 2024-07-31 balance sheet.
    depr_rate_fwd = policies.depreciation_rate[:, -horizon:, :]
    base_ppe = 1.0
    # if "2025-04-30" in bal.columns:
    #     bs_col = bal["2025-04-30"]
    #     for idx, val in bs_col.items():
    #         if idx.strip().lower() == "net ppe":
    #             base_ppe = float(val)
    #             break
    depr = depr_rate_fwd * base_ppe

    gamma = driver_params.get("gamma_capex_growth", 0.0)
    # Sales deltas: dsales_t = sales_t - sales_{t-1}, with dsales_0 relative
    # to the last historical point so it is nontrivial at the first step.
    if sales_hist.size > 0:
        last_hist_val = float(sales_hist[-1])
    else:
        last_hist_val = 0.0
    sales_full = np.concatenate([[last_hist_val], sales_fut.reshape(-1)], axis=0)
    dsales_np = sales_full[1:] - sales_full[:-1]
    dsales = tf.convert_to_tensor(dsales_np.reshape(1, horizon, 1), dtype=tf.float32)

    capex = depr # + gamma * tf.nn.relu(dsales)

    return DriversWMT(
        sales=sales_tf,
        cogs=cogs_tf,
        capex=capex,
    )


def simulate_aggregate_invest(
    cf: pd.DataFrame,
    driver_params: Dict[str, float],
    horizon: int,
) -> tf.Tensor:
    """Simulate a simple forward aggregate_invest path.

    For now we use a very lightweight heuristic consistent with
    estimate_investing_sensitivity: treat aggregate_invest as a
    modest multiple of recent capex. This keeps the sign convention
    (negative = cash out for acquisitions) and allows goodwill/other
    NCA and leases to move in forecast.
    """

    # Locate capex history from CF to infer a typical scale
    capex_row = None
    for r in cf.index:
        if "capital expenditure" in r.lower():
            capex_row = r
            break
    if capex_row is None:
        # No capex history; assume zero aggregate_invest
        return tf.zeros([1, horizon, 1], dtype=tf.float32)

    capex_hist = -cf.loc[capex_row].to_numpy(dtype=np.float32)  # positive outflow
    # Use median of positive capex as typical scale
    capex_pos = capex_hist[capex_hist > EPS]
    if capex_pos.size == 0:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)
    capex_scale = float(np.median(capex_pos))

    # Heuristic: assume net acquisitions are a small fraction of capex.
    # We don't yet estimate this gamma separately, so pick a conservative
    # scale based on beta2_net_invest magnitude if available.
    beta2 = float(driver_params.get("beta2_net_invest", 0.0))
    # Map beta2 into a rough share in [-0.5, 0.0] if negative, else small.
    if beta2 < 0.0:
        gamma_inv = max(beta2, -0.5)
    else:
        gamma_inv = -0.1

    # Build a flat path at this scale (negative: cash outflow for acquisitions)
    agg_vals = np.full((1, horizon, 1), gamma_inv * capex_scale, dtype=np.float32)
    return tf.convert_to_tensor(agg_vals, dtype=tf.float32)


def simulate_aoci_driver(
    prev: PrevStateWMT,
    driver_params: Dict[str, float],
    horizon: int,
) -> tf.Tensor:
    """Build a simple forward change_in_aoci path using estimated drift.

    We use a constant per-period drift (in currency units) equal to the
    historical average ΔAOCI estimated in estimate_wmt. This is intended as a
    minimal heuristic; scenario-specific overrides can later replace it.
    """

    drift = float(driver_params.get("aoci_drift", 0.0))
    if drift == 0.0:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)
    arr = np.full((1, horizon, 1), drift, dtype=np.float32)
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def _estimate_ar1_from_series(values: np.ndarray) -> tuple[float, float]:
    """Estimate a simple AR(1): x_t = mu + phi * (x_{t-1} - mu).

    We use a very small sample (last few quarters), so estimation is kept
    deliberately simple and robust:

    - ``mu`` is the mean of the input values.
    - ``phi`` is estimated by OLS on demeaned series and clipped to [-0.95, 0.95].

    When there are fewer than 2 non-NaN points, we fall back to ``phi = 0``.
    """

    x = values[~np.isnan(values)].astype(np.float32)
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x[0]), 0.0

    mu = float(x.mean())
    xd = x - mu
    x_tm1 = xd[:-1]
    x_t = xd[1:]
    denom = np.dot(x_tm1, x_tm1)
    if abs(denom) < EPS:
        phi = 0.0
    else:
        phi = float(np.dot(x_tm1, x_t) / denom)
    # Clip to keep the process stationary and avoid explosions.
    phi = float(np.clip(phi, -0.95, 0.95))
    return mu, phi


def simulate_deferred_and_other_non_cash(
    cf: pd.DataFrame,
    horizon: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Simulate deferred income tax and other non-cash items with AR(1).

    Both series are modeled as:

    .. math::

        x_t = \mu + \phi (x_{t-1} - \mu)

    where ``mu`` and ``phi`` are estimated from the last few historical
    quarters of the corresponding cash-flow rows. This keeps the series
    free to be positive or negative and does not tie them mechanically
    to net income or taxes.
    """

    # Locate rows in cash flow
    other_nc_row = None
    for r in cf.index:
        if "other non cash items" in r.lower():
            other_nc_row = r
            break
    deferred_row = None
    for r in cf.index:
        if "deferred income tax" in r.lower() or "deferred tax" in r.lower():
            deferred_row = r
            break

    def _hist_values(row_name: str | None) -> np.ndarray:
        if row_name is None:
            return np.zeros(0, dtype=np.float32)
        series = cf.loc[row_name]
        # Use the same four-quarter window as in training when available.
        hist_cols = [
            "2024-07-31",
            "2024-10-31",
            "2025-01-31",
            "2025-04-30",
        ]
        vals: list[float] = []
        for c in hist_cols:
            if c in series.index and not pd.isna(series[c]):
                vals.append(float(series[c]))
        if not vals:
            # Fallback: all non-NaN values as stored (latest to oldest)
            vals = [float(v) for v in series.to_numpy(dtype=np.float32) if not np.isnan(v)]
        return np.array(vals, dtype=np.float32)

    other_nc_hist = _hist_values(other_nc_row)
    deferred_hist = _hist_values(deferred_row)

    mu_other, phi_other = _estimate_ar1_from_series(other_nc_hist)
    mu_def, phi_def = _estimate_ar1_from_series(deferred_hist)

    def _simulate_path(last_hist: np.ndarray, mu: float, phi: float) -> np.ndarray:
        # If we have at least one historical value, start from its last point;
        # otherwise, start directly from mu.
        if last_hist.size > 0:
            x_prev = float(last_hist[-1])
        else:
            x_prev = mu
        path: list[float] = []
        for _ in range(horizon):
            x_next = mu + phi * (x_prev - mu)
            path.append(float(x_next))
            x_prev = x_next
        return np.array(path, dtype=np.float32).reshape(1, horizon, 1)

    other_nc_path = _simulate_path(other_nc_hist, mu_other, phi_other)
    deferred_path = _simulate_path(deferred_hist, mu_def, phi_def)

    return (
        tf.convert_to_tensor(deferred_path, dtype=tf.float32),
        tf.convert_to_tensor(other_nc_path, dtype=tf.float32),
    )


def simulate_change_in_accrued_expenses(cf: pd.DataFrame, horizon: int) -> tf.Tensor:
    """Simulate change_in_accrued_expenses using a simple AR(1) on CF deltas.

    We use the vendor cash-flow row "Change In Accrued Expense" as the
    historical series and fit an AR(1) process directly on that delta:

        ΔA_t = μ + φ (ΔA_{t-1} - μ)

    with μ the sample mean over the last few quarters and φ estimated by OLS
    on the demeaned series (clipped to [-0.95, 0.95]). This keeps the sign and
    magnitude behavior consistent with history without tying it to macro/NI.
    """

    acc_row = None
    for r in cf.index:
        if "change in accrued expense" in r.lower():
            acc_row = r
            break

    if acc_row is None:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)

    series = cf.loc[acc_row]
    hist_cols = [
        "2024-07-31",
        "2024-10-31",
        "2025-01-31",
        "2025-04-30",
    ]
    vals: list[float] = []
    for c in hist_cols:
        if c in series.index and not pd.isna(series[c]):
            vals.append(float(series[c]))
    if not vals:
        vals = [float(v) for v in series.to_numpy(dtype=np.float32) if not np.isnan(v)]
    hist = np.array(vals, dtype=np.float32)

    mu, phi = _estimate_ar1_from_series(hist)

    if hist.size > 0:
        prev = float(hist[-1])
    else:
        prev = mu
    path: list[float] = []
    for _ in range(horizon):
        nxt = mu + phi * (prev - mu)
        path.append(float(nxt))
        prev = nxt

    arr = np.array(path, dtype=np.float32).reshape(1, horizon, 1)
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def simulate_net_common_stock_issuance(cf: pd.DataFrame, horizon: int) -> tf.Tensor:
    """Simulate net_common_stock_issuance using an AR(1) on the CF line.

    We use the vendor cash-flow row "Net Common Stock Issuance" as the
    historical series and fit an AR(1):

        x_t = μ + φ (x_{t-1} - μ)

    with μ the sample mean over the last few quarters and φ estimated by OLS
    on the demeaned series (clipped to [-0.95, 0.95]). This allows the series
    to remain negative on average (net buybacks) or positive (net issuance).
    """

    row = None
    for r in cf.index:
        if "net common stock issuance" in r.lower():
            row = r
            break

    if row is None:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)

    series = cf.loc[row]
    hist_cols = [
        "2024-07-31",
        "2024-10-31",
        "2025-01-31",
        "2025-04-30",
    ]
    vals: list[float] = []
    for c in hist_cols:
        if c in series.index and not pd.isna(series[c]):
            vals.append(float(series[c]))
    if not vals:
        vals = [float(v) for v in series.to_numpy(dtype=np.float32) if not np.isnan(v)]
    hist = np.array(vals, dtype=np.float32)

    mu, phi = _estimate_ar1_from_series(hist)

    if hist.size > 0:
        prev = float(hist[-1])
    else:
        prev = mu
    path: list[float] = []
    for _ in range(horizon):
        nxt = mu + phi * (prev - mu)
        path.append(float(nxt))
        prev = nxt

    arr = np.array(path, dtype=np.float32).reshape(1, horizon, 1)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

def simulate_cash_dividends_paid(
    fin: pd.DataFrame,
    cf: pd.DataFrame,
    policies: PoliciesWMT,
    horizon: int,
) -> tf.Tensor:
    """Simulate forecast cash_dividends_paid using a lagged payout-on-net-income rule.

    - For history, we use the last available "Cash Dividends Paid" as the anchor.
    - For forecast, we approximate declarations as payout * net_income and
      map them to cash one quarter later via a simple smoothing rule.

    This function does not change the deterministic layer logic; it only
    generates the driver series for forecast periods.
    """
    # Locate Net Income and Cash Dividends Paid rows
    ni_row = None
    for r in fin.index:
        if "net income from continuing operations" in r.lower() or "net income" in r.lower():
            ni_row = r
            break
    if ni_row is None:
        raise ValueError("Net Income row not found in financials.")

    cash_div_row = None
    for r in cf.index:
        if "cash dividends paid" in r.lower():
            cash_div_row = r
            break
    if cash_div_row is None:
        raise ValueError("Cash Dividends Paid row not found in cash flow.")

    ni_series = fin.loc[ni_row]
    cd_series = cf.loc[cash_div_row]

    # Historical window aligned with training (same as in training/estimation scripts)
    hist_cols = [
        "2024-07-31",
        "2024-10-31",
        "2025-01-31",
        "2025-04-30",
    ]
    ni_hist = []
    for c in hist_cols:
        if c in ni_series.index and not pd.isna(ni_series[c]):
            ni_hist.append(float(ni_series[c]))
    ni_hist_arr = np.array(ni_hist, dtype=np.float32) if ni_hist else np.array([], dtype=np.float32)

    if ni_hist_arr.size == 0:
        ni_mean = 0.0
    else:
        ni_mean = float(ni_hist_arr.mean())

    # Last historical cash dividends paid (anchor)
    last_cd = None
    for c in hist_cols[::-1]:  # search from most recent backwards
        if c in cd_series.index and not pd.isna(cd_series[c]):
            last_cd = float(cd_series[c])
            break
    if last_cd is None:
        # Fallback: scan all columns for last non-NaN
        for c in cd_series.index:
            v = cd_series[c]
            if not pd.isna(v):
                last_cd = float(v)
                break
    if last_cd is None:
        last_cd = 0.0

    # Use a single scalar payout based on policies (take last available)
    # policies.payout_ratio shape: [B, T, 1]; here we assume B=1 and use last time step.
    payout_arr = policies.payout_ratio.numpy()
    payout_scalar = float(payout_arr[0, -1, 0]) if payout_arr.size > 0 else 0.0

    # Simple parameters for smoothing
    alpha = 0.2  # blend towards historical mean net income
    lam = 1.0    # how strongly to follow lagged declaration (1.0 = full lag)

    # For forecast we don't have future net income yet; approximate using ni_mean
    # and assume flat path at that level. This keeps the rule simple and
    # purely driver-based for now.
    forecast_cd = []
    prev_cd = max(last_cd, 0.0)
    prev_decl = payout_scalar * max(ni_mean, 0.0)
    for _ in range(horizon):
        # Smoothed declaration proxy (could be extended later to use model NI)
        decl = payout_scalar * max((1.0 - alpha) * ni_mean + alpha * ni_mean, 0.0)
        cash_div = (1.0 - lam) * prev_cd + lam * prev_decl
        # Clip to avoid negatives
        cash_div = max(cash_div, 0.0)
        forecast_cd.append(cash_div)
        prev_cd = cash_div
        prev_decl = decl

    arr = np.array(forecast_cd, dtype=np.float32).reshape(1, horizon, 1)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

def simulate_change_in_minority_interest(
    bal: pd.DataFrame,
    horizon: int,
) -> tf.Tensor:
    """Simulate change_in_minority_interest using historical-average BS deltas.

    We take the Minority Interest row from the balance sheet, compute
    ΔMinority_t = M_t - M_{t-1}, then use the time-average of these deltas
    as a flat forecast driver.
    """
    row = None
    for r in bal.index:
        if r.strip().lower() == "minority interest":
            row = r
            break
    if row is None:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)

    series = bal.loc[row].to_numpy(dtype=np.float32)  # as stored: latest→oldest in CSV
    # Keep only non-NaNs
    series = series[~np.isnan(series)]
    if series.size < 2:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)

    # Compute deltas along time axis: Δ_t = M_t - M_{t-1}
    deltas = series[1:] - series[:-1]
    if deltas.size == 0:
        return tf.zeros([1, horizon, 1], dtype=tf.float32)

    avg_delta = float(np.nanmean(deltas))

    # Simple safety clip: at most 5% of last level per quarter
    last_level = float(series[-1])
    cap = 0.05 * abs(last_level) if last_level != 0.0 else abs(avg_delta)
    if cap > 0.0:
        avg_delta = float(np.clip(avg_delta, -cap, cap))

    arr = np.full((1, horizon, 1), avg_delta, dtype=np.float32)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

def main(horizon_quarters: int = 1) -> None:
    policies = load_policies()
    driver_params = load_driver_params()
    # These reads are kept for potential future scenario design but no longer
    # drive the choice of "last" column directly.
    bal = pd.read_csv(BAL_PATH, index_col=0)
    fin = pd.read_csv(FIN_PATH, index_col=0)
    cf = pd.read_csv(CF_PATH, index_col=0)

    prev = load_last_prev_state()
    drivers = simulate_sales_and_capex(bal, fin, policies, driver_params, horizon_quarters)
    # Simulate cash dividends paid driver for forecast horizon
    cash_div = simulate_cash_dividends_paid(fin, cf, policies, horizon_quarters)
    # Simple AOCI driver using constant drift
    change_in_aoci = simulate_aoci_driver(prev, driver_params, horizon_quarters)
    # Simple aggregate investing driver so goodwill/other NCA and leases move
    aggregate_invest = simulate_aggregate_invest(cf, driver_params, horizon_quarters)
    # AR(1) simulation for deferred income tax and other non-cash items
    deferred_income_tax, other_non_cash_items = simulate_deferred_and_other_non_cash(cf, horizon_quarters)
    # AR(1) simulation for change in accrued expenses
    change_in_accrued_expenses = simulate_change_in_accrued_expenses(cf, horizon_quarters)
    # AR(1) simulation for net common stock issuance (issuance positive, buybacks negative); deprecated;
    net_common_stock_issuance = simulate_net_common_stock_issuance(cf, horizon_quarters)
    change_in_minority_interest = simulate_change_in_minority_interest(bal, horizon_quarters)

    drivers = DriversWMT(
        sales=drivers.sales,
        cogs=drivers.cogs,
        capex=drivers.capex,
        cash_dividends_paid=cash_div,
        change_in_aoci=change_in_aoci,
        aggregate_invest=aggregate_invest,
        change_in_accrued_expenses=change_in_accrued_expenses,
        deferred_income_tax=deferred_income_tax,
        other_non_cash_items=other_non_cash_items,
        net_common_stock_issuance=net_common_stock_issuance,
        change_in_minority_interest=change_in_minority_interest,
    )

    layer = StructuralLayer()
    stmts = layer.call(drivers, policies, prev)

    # Headline forecast numbers
    sales_f = stmts.sales.numpy()[0, :, 0]
    cash_f = stmts.cash.numpy()[0, :, 0]
    equity_f = stmts.equity.numpy()[0, :, 0]
    # Relative accounting identity gap over the forecast horizon
    assets = stmts.assets
    liab_eq = stmts.liab_plus_equity
    denom = tf.maximum(tf.maximum(tf.abs(assets), tf.abs(liab_eq)), tf.constant(1e-6, dtype=assets.dtype))
    rel_gap = (assets - liab_eq) / denom
    rel_gap_rmse = tf.sqrt(tf.reduce_mean(tf.square(rel_gap)))
    print("Forecast horizon (quarters):", horizon_quarters)
    # Assets side
    cash_arr = stmts.cash.numpy()[0, :, 0]
    st_inv_arr = stmts.st_investments.numpy()[0, :, 0]
    inventory_arr = stmts.inventory.numpy()[0, :, 0]
    oca_arr = stmts.other_current_assets.numpy()[0, :, 0]
    goodwill_arr = stmts.goodwill_intangibles.numpy()[0, :, 0]
    net_ppe_arr = stmts.net_ppe.numpy()[0, :, 0]
    assets_arr = assets.numpy()[0, :, 0]
    print("Cash forecast:", cash_arr)
    print("Short-term investments forecast:", st_inv_arr)
    print("Inventory forecast:", inventory_arr)
    print("Other Current Assets forecast:", oca_arr)
    print("Goodwill & Intangibles forecast:", goodwill_arr)
    print("Net PPE forecast:", net_ppe_arr)
    print("Assets forecast:", assets_arr)
    # Liabilities side
    st_debt_arr = stmts.st_debt.numpy()[0, :, 0]
    cur_lease_arr = stmts.current_capital_lease_obligation.numpy()[0, :, 0]
    lt_debt_arr = stmts.lt_debt.numpy()[0, :, 0]
    lt_lease_arr = stmts.long_term_capital_lease_obligation.numpy()[0, :, 0]
    ap_arr = stmts.ap.numpy()[0, :, 0]
    accrued_arr = stmts.accrued_expenses.numpy()[0, :, 0]
    tax_payable_arr = stmts.tax_payable.numpy()[0, :, 0]
    div_payable_arr = stmts.dividends_payable.numpy()[0, :, 0]
    other_ncl_arr = stmts.other_non_current_liabilities.numpy()[0, :, 0]
    liab_only_arr = (liab_eq - stmts.equity).numpy()[0, :, 0]
    print("Short-term debt forecast:", st_debt_arr)
    print("Current capital lease obligation forecast:", cur_lease_arr)
    print("Long-term debt forecast:", lt_debt_arr)
    print("Long-term capital lease obligation forecast:", lt_lease_arr)
    print("Accounts payable forecast:", ap_arr)
    print("Accrued expenses forecast:", accrued_arr)
    print("Tax payable forecast:", tax_payable_arr)
    print("Dividends payable forecast:", div_payable_arr)
    print("Other Non-Current Liabilities forecast:", other_ncl_arr)
    print("Liabilities forecast:", liab_only_arr)
    # Equity block
    re_arr = stmts.retained_earnings.numpy()[0, :, 0]
    pic_arr = stmts.paid_in_capital.numpy()[0, :, 0]
    cap_stock_arr = stmts.capital_stock.numpy()[0, :, 0]
    aoci_arr = stmts.aoci.numpy()[0, :, 0]
    minority_arr = stmts.minority_interest.numpy()[0, :, 0]
    liab_eq_arr = liab_eq.numpy()[0, :, 0]
    print("Retained Earnings forecast:", re_arr)
    print("Paid-in Capital forecast:", pic_arr)
    print("Capital Stock forecast:", cap_stock_arr)
    print("Accumulated Other Comprehensive Income forecast:", aoci_arr)
    print("Minority Interest forecast:", minority_arr)
    print("Equity forecast:", equity_f)
    print("Liabilities + Equity forecast:", liab_eq_arr)
    print("Relative accounting identity RMSE:", float(rel_gap_rmse.numpy()))
    # Income statement flows
    cogs_arr = stmts.cogs.numpy()[0, :, 0]
    opex_arr = stmts.opex.numpy()[0, :, 0]
    depr_arr = stmts.depreciation.numpy()[0, :, 0]
    ebit_arr = stmts.ebit.numpy()[0, :, 0]
    ni_arr = stmts.net_income.numpy()[0, :, 0]
    print("Sales forecast:", sales_f)
    print("Cogs forecast:", cogs_arr)
    print("Opex forecast:", opex_arr)
    print("Depreciation forecast:", depr_arr)
    print("EBIT forecast:", ebit_arr)
    print("Net Income forecast:", ni_arr)
    # Cash flow statement flows
    capex_arr = stmts.capex.numpy()[0, :, 0]
    print("Capital Expenditures forecast:", capex_arr)

    # Persist forecast as CSV for downstream analysis
    periods = np.arange(1, horizon_quarters + 1)
    data = {
        "period": periods,
        "cash": cash_arr,
        "st_investments": st_inv_arr,
        "inventory": inventory_arr,
        "other_current_assets": oca_arr,
        "goodwill_intangibles": goodwill_arr,
        "net_ppe": net_ppe_arr,
        "assets": assets_arr,
        "st_debt": st_debt_arr,
        "current_capital_lease_obligation": cur_lease_arr,
        "lt_debt": lt_debt_arr,
        "long_term_capital_lease_obligation": lt_lease_arr,
        "ap": ap_arr,
        "accrued_expenses": accrued_arr,
        "tax_payable": tax_payable_arr,
        "dividends_payable": div_payable_arr,
        "other_non_current_liabilities": other_ncl_arr,
        "liabilities": liab_only_arr,
        "retained_earnings": re_arr,
        "paid_in_capital": pic_arr,
        "capital_stock": cap_stock_arr,
        "aoci": aoci_arr,
        "minority_interest": minority_arr,
        "equity": equity_f,
        "liab_plus_equity": liab_eq_arr,
        "sales": sales_f,
        "cogs": cogs_arr,
        "opex": opex_arr,
        "depreciation": depr_arr,
        "ebit": ebit_arr,
        "net_income": ni_arr,
        "capex": capex_arr,
        "relative_identity_rmse": float(rel_gap_rmse.numpy()),
    }
    df = pd.DataFrame(data)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "wmt_forecast_quarterly.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved forecast to {out_path}")


if __name__ == "__main__":
    main()
