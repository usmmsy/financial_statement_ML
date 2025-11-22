"""Train/estimate WMT-specific policies and driver params from quarterly CSVs.

This is a first-pass, low-variance estimator using 4 quarters of data as training
and (optionally) the most recent quarter as a rough validation slice.

Outputs:
- npz file with forward PoliciesWMT tensors for a 1-quarter forecast horizon.
- npz file with basic scalar driver parameters (currently capex growth coeff).

This script is intentionally simple; refine formulas as needed.
"""
from __future__ import annotations
import pathlib
from typing import Dict, Any

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

from wmt_bs_forecaster.types_wmt import PoliciesWMT

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "retail_csv" / "WMT_quarterly"
OUT_DIR = ROOT / "data" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAL_PATH = DATA_DIR / "WMT_quarterly_balance_sheet.csv"
FIN_PATH = DATA_DIR / "WMT_quarterly_financials.csv"
CF_PATH = DATA_DIR / "WMT_quarterly_cash_flow.csv"

EPS = 1e-8


# --- Local estimation helpers (self-contained) ---

def constant_tensor(value: float, T: int) -> tf.Tensor:
    """Create a [1, T, 1] constant tensor.

    This mirrors the shape convention used by PoliciesWMT/DriversWMT.
    """
    arr = np.full((1, T, 1), float(value), dtype=np.float32)
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def _normalize_label(text: str) -> str:
    """Normalize a row label or phrase for robust matching.

    Lowercase and collapse internal whitespace so that minor spacing
    differences don't break exact phrase matching.
    """
    return " ".join(text.lower().split())


def _find_row(df: pd.DataFrame, phrase: str) -> str | None:
    """Return the row label that best matches *phrase*.

    Strategy (in order):
    1) Exact normalized match (case/whitespace-insensitive).
    2) Unique normalized substring match; if multiple candidates, give up.
    """
    phrase_norm = _normalize_label(phrase)
    # 1) Exact normalized match
    for idx in df.index:
        if _normalize_label(idx) == phrase_norm:
            return idx

    # 2) Unique substring match as a soft fallback
    candidates = [idx for idx in df.index if phrase_norm in _normalize_label(idx)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def learn_gross_margin(fin: pd.DataFrame) -> tf.Tensor:
    """Estimate gross margin policy as a constant [1,T,1] series.

    gross_margin_t = 1 - COGS_t / max(revenue_t, eps)
    and we use the time-average as a low-variance estimate.
    """
    rev_row = _find_row(fin, "Total Revenue")
    # COGS is reported as Cost Of Revenue in WMT CSVs
    cogs_row = _find_row(fin, "Cost Of Revenue")
    T = fin.shape[1]
    if rev_row is None or cogs_row is None:
        # Fallback to a plausible retail margin if rows not found
        return constant_tensor(0.25, T)
    sales = fin.loc[rev_row].to_numpy(dtype=np.float32)
    cogs = fin.loc[cogs_row].to_numpy(dtype=np.float32)
    margin_series = 1.0 - cogs / np.maximum(sales, EPS)
    margin = float(np.nanmean(margin_series))
    margin = float(np.clip(margin, 0.0, 0.9))
    return constant_tensor(margin, T)


def pseudo_learn_opex_ratio(fin: pd.DataFrame) -> tf.Tensor:
    """Estimate opex / sales as a constant policy [1,T,1].

    Prefer SG&A; if missing/empty for some quarters, fall back to Operating Expense.
    """
    T = fin.shape[1]
    rev_row = _find_row(fin, "Total Revenue")
    if rev_row is None:
        return constant_tensor(0.2, T)

    # Primary: SG&A; if missing or NaN for a given quarter, fall back to Operating Expense
    # sgna_row = _find_row(fin, "Selling General And Administrative")
    opex_row = _find_row(fin, "Operating Expense")
    if opex_row is None:
        return constant_tensor(0.2, T)

    sales = fin.loc[rev_row].to_numpy(dtype=np.float32)
    
    opex = fin.loc[opex_row].to_numpy(dtype=np.float32)

    ratio_series = opex / sales
    ratio = float(np.nanmean(ratio_series))
    ratio = float(np.clip(ratio, 0.0, 0.8))
    return constant_tensor(ratio, T), opex, sales

def learn_opex_ratio(fin: pd.DataFrame) -> tf.Tensor:
    """Estimate opex / sales as a constant policy [1,T,1].

    Prefer SG&A; if missing/empty for some quarters, fall back to Operating Expense.
    """
    T = fin.shape[1]
    rev_row = _find_row(fin, "Total Revenue")
    if rev_row is None:
        return constant_tensor(0.2, T)

    # Primary: SG&A; if missing or NaN for a given quarter, fall back to Operating Expense
    # sgna_row = _find_row(fin, "Selling General And Administrative")
    opex_row = _find_row(fin, "Operating Expense")
    if opex_row is None:
        return constant_tensor(0.2, T)

    sales = fin.loc[rev_row].to_numpy(dtype=np.float32)
    
    opex = fin.loc[opex_row].to_numpy(dtype=np.float32)

    ratio_series = opex / sales
    ratio = float(np.nanmean(ratio_series))
    ratio = float(np.clip(ratio, 0.0, 0.8))
    return constant_tensor(ratio, T)


def learn_depreciation_rate(fin: pd.DataFrame, ppe_prev: np.ndarray) -> tf.Tensor:
    """Estimate a single depreciation rate over beginning net PPE.

    rate_t = depreciation_t / max(PPE_beg_t, eps), averaged over time.
    """
    T = fin.shape[1]
    depr_row = _find_row(fin, "Depreciation")
    if depr_row is None or ppe_prev.size != T:
        return constant_tensor(0.03, T)
    depr = fin.loc[depr_row].to_numpy(dtype=np.float32)
    rate_series = depr / np.maximum(ppe_prev.astype(np.float32), EPS)
    rate = float(np.nanmean(rate_series))
    rate = float(np.clip(rate, 0.0, 0.5))
    return constant_tensor(rate, T)


def learn_working_capital_days(bal: pd.DataFrame, fin: pd.DataFrame) -> Dict[str, tf.Tensor]:
    """Estimate DSO/DPO/DIO days from balances and sales/COGS.

    days ≈ period_days * balance / max(flow, eps), averaged over time.
    """
    T = fin.shape[1]
    period_days = 365.0 / 4.0  # quarterly approximation

    rev_row = _find_row(fin, "Total Revenue")
    # COGS as Cost Of Revenue in WMT CSVs
    cogs_row = _find_row(fin, "Cost Of Revenue")
    ar_row = _find_row(bal, "Accounts Receivable")
    ap_row = _find_row(bal, "Accounts Payable")
    inv_row = _find_row(bal, "Inventory")

    if rev_row is None or cogs_row is None or ar_row is None or ap_row is None or inv_row is None:
        # Fallback to typical retail WC days
        return {
            "dso_days": constant_tensor(5.0, T),
            "dpo_days": constant_tensor(40.0, T),
            "dio_days": constant_tensor(40.0, T),
        }

    sales = fin.loc[rev_row].to_numpy(dtype=np.float32)
    cogs = fin.loc[cogs_row].to_numpy(dtype=np.float32)
    ar = bal.loc[ar_row].to_numpy(dtype=np.float32)
    ap = bal.loc[ap_row].to_numpy(dtype=np.float32)
    inv = bal.loc[inv_row].to_numpy(dtype=np.float32)

    dso_series = period_days * ar / np.maximum(sales, EPS)
    dpo_series = period_days * ap / np.maximum(cogs, EPS)
    dio_series = period_days * inv / np.maximum(cogs, EPS)

    dso = float(np.nanmean(dso_series))
    dpo = float(np.nanmean(dpo_series))
    dio = float(np.nanmean(dio_series))

    dso = float(np.clip(dso, 0.0, 120.0))
    dpo = float(np.clip(dpo, 0.0, 180.0))
    dio = float(np.clip(dio, 0.0, 200.0))

    return {
        "dso_days": constant_tensor(dso, T),
        "dpo_days": constant_tensor(dpo, T),
        "dio_days": constant_tensor(dio, T),
    }


def learn_tax_rate(fin: pd.DataFrame) -> tf.Tensor:
    """Estimate an effective tax rate from income before income taxes.

    tax_rate_t = tax_expense_t / max(EBT_t_positive, eps), averaged over time.
    """
    T = fin.shape[1]
    # Use Tax Provision and Pretax Income from WMT financials
    tax_row = _find_row(fin, "Tax Provision")
    ebt_row = _find_row(fin, "Pretax Income")
    if tax_row is None or ebt_row is None:
        return constant_tensor(0.21, T)
    tax = fin.loc[tax_row].to_numpy(dtype=np.float32)
    ebt = fin.loc[ebt_row].to_numpy(dtype=np.float32)
    ebt_pos = np.maximum(ebt, 0.0)
    mask = ebt_pos > EPS
    if not np.any(mask):
        return constant_tensor(0.21, T)
    rate_series = tax[mask] / np.maximum(ebt_pos[mask], EPS)
    rate = float(np.nanmean(rate_series))
    rate = float(np.clip(rate, 0.0, 0.5))
    return constant_tensor(rate, T)


def learn_payout_ratio(fin: pd.DataFrame, div_declared: np.ndarray | None) -> tf.Tensor:
    """Estimate payout_ratio ≈ dividends_declared / net_income.

    If dividends are missing, fall back to a modest constant.
    """
    T = fin.shape[1]
    ni_row = _find_row(fin, "Net Income")
    if ni_row is None:
        return constant_tensor(0.3, T)

    net_inc = fin.loc[ni_row].to_numpy(dtype=np.float32)
    if div_declared is None or div_declared.size != net_inc.size:
        return constant_tensor(0.3, T)

    ni_pos = np.maximum(net_inc, 0.0)
    mask = ni_pos > EPS
    if not np.any(mask):
        return constant_tensor(0.3, T)
    ratio_series = div_declared[mask].astype(np.float32) / np.maximum(ni_pos[mask], EPS)
    ratio = float(np.nanmean(ratio_series))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return constant_tensor(ratio, T)


def load_wmt_quarterlies() -> Dict[str, pd.DataFrame]:
    bal = pd.read_csv(BAL_PATH, index_col=0)
    fin = pd.read_csv(FIN_PATH, index_col=0)
    cf = pd.read_csv(CF_PATH, index_col=0)
    # Align columns (time axis) by intersecting date labels and sorting by date ascending.
    common_cols = list(sorted(set(bal.columns) & set(fin.columns) & set(cf.columns)))
    bal = bal[common_cols]
    fin = fin[common_cols]
    cf = cf[common_cols]
    # orient so columns represent time, but for estimation we'll just treat them as sequence
    return {"bal": bal, "fin": fin, "cf": cf}


def extract_ppe_prev(bal: pd.DataFrame) -> np.ndarray:
    """Approximate beginning-of-period net PPE as previous period's end PPE.

    For the very first period, reuse the first value as a crude approximation.
    """
    # Find Net PPE column flexibly
    ppe_col = None
    for c in bal.index:
        if "net ppe" in c.lower():
            ppe_col = c
            break
    if ppe_col is None:
        raise ValueError("Cannot locate Net PPE row in balance sheet CSV.")
    ppe_end = bal.loc[ppe_col].to_numpy(dtype=np.float32)
    if ppe_end.size < 2:
        return ppe_end
    ppe_beg = np.concatenate([[ppe_end[0]], ppe_end[:-1]])
    return ppe_beg


def learn_basic_policies(bal: pd.DataFrame, fin: pd.DataFrame, cf: pd.DataFrame) -> Dict[str, tf.Tensor]:
    # Each DF has columns = quarters; we treat them as time axis
    # Use all available quarters here; caller can later choose train/validation split.
    T = fin.shape[1]
    gross_margin = learn_gross_margin(fin)
    opex_ratio = learn_opex_ratio(fin)
    ppe_prev = extract_ppe_prev(bal)
    depr_rate = learn_depreciation_rate(fin, ppe_prev)
    wc_days = learn_working_capital_days(bal, fin)
    tax_rate = learn_tax_rate(fin)
    # Dividends declared: approximate from cash flow dividends paid (no lag modeling here)
    div_paid_row = None
    for r in cf.index:
        if "cash dividends paid" in r.lower():
            div_paid_row = r
            break
    div_declared = cf.loc[div_paid_row].to_numpy(dtype=np.float32) if div_paid_row is not None else None
    payout_ratio = learn_payout_ratio(fin, div_declared)
    # Simple constant paths for macro & liquidity for now
    inflation = constant_tensor(0.02 / 4.0, T)  # 2% annualized
    real_st_rate = constant_tensor(0.0, T)
    real_lt_rate = constant_tensor(0.01, T)
    min_cash_ratio = constant_tensor(0.02, T)
    lt_share_for_capex = constant_tensor(0.5, T)
    st_invest_spread = constant_tensor(0.0, T)
    debt_spread = constant_tensor(0.02 / 4.0, T)
    return {
        "inflation": inflation,
        "real_st_rate": real_st_rate,
        "real_lt_rate": real_lt_rate,
        "tax_rate": tax_rate,
        "payout_ratio": payout_ratio,
        "min_cash_ratio": min_cash_ratio,
        "lt_share_for_capex": lt_share_for_capex,
        "st_invest_spread": st_invest_spread,
        "debt_spread": debt_spread,
        "dso_days": wc_days["dso_days"],
        "dpo_days": wc_days["dpo_days"],
        "dio_days": wc_days["dio_days"],
        "opex_ratio": opex_ratio,
        "depreciation_rate": depr_rate,
        "gross_margin": gross_margin,
    }


def estimate_capex_growth_coeff(fin: pd.DataFrame, cf: pd.DataFrame) -> float:
    """Estimate a simple capex growth coefficient gamma using history.

    gamma ~= median( (capex - depreciation) / max(Δsales_positive, eps) ).
    """
    # Locate relevant rows
    capex_row = None
    for r in cf.index:
        if "capital expenditure" in r.lower():
            capex_row = r
            break
    if capex_row is None:
        return 0.0
    capex = -cf.loc[capex_row].to_numpy(dtype=np.float32)  # cash flow is negative for capex

    rev_row = None
    for r in fin.index:
        if "total revenue" in r.lower():
            rev_row = r
            break
    if rev_row is None:
        return 0.0
    sales = fin.loc[rev_row].to_numpy(dtype=np.float32)
    # Depreciation
    depr_row = None
    for r in fin.index:
        if "depreciation" in r.lower():
            depr_row = r
            break
    if depr_row is None:
        return 0.0
    depr = fin.loc[depr_row].to_numpy(dtype=np.float32)

    # compute positive sales deltas
    if sales.size < 2:
        return 0.0
    dsales = np.maximum(sales[1:] - sales[:-1], 0.0)
    extra_capex = capex[1:] - depr[1:]
    mask = dsales > EPS
    if not np.any(mask):
        return 0.0
    gamma_samples = extra_capex[mask] / np.maximum(dsales[mask], EPS)
    if gamma_samples.size == 0:
        return 0.0
    gamma = float(np.median(gamma_samples))
    return max(0.0, gamma)

def estimate_aoci_drift(bal: pd.DataFrame) -> float:
    """Estimate a simple AOCI drift parameter from history.

    We use the vendor "Gains Losses Not Affecting Retained Earnings" line as
    total AOCI and compute per-period changes ΔAOCI_t. As a very lightweight
    heuristic, we return the time-average ΔAOCI (in absolute currency units).

    This scalar can be used by forecasting code to build a baseline
    change_in_aoci path (e.g., constant drift) when no scenario is provided.
    """

    aoci_row = _find_row(bal, "Gains Losses Not Affecting Retained Earnings")
    if aoci_row is None:
        return 0.0
    series = bal.loc[aoci_row].to_numpy(dtype=np.float32)
    if series.size < 2:
        return 0.0
    deltas = series[1:] - series[:-1]
    if deltas.size == 0:
        return 0.0
    drift = float(np.nanmean(deltas))
    # Leave sign as-is; AOCI can be positive or negative. No clipping.
    return float(drift)


def estimate_investing_sensitivity(bal: pd.DataFrame, cf: pd.DataFrame) -> Dict[str, float]:
    """Estimate simple coefficients linking aggregate investing CF to goodwill/other NCA.

    This is deliberately low-variance and coarse:

    - premium_ratio_goodwill ~= median( ΔGoodwill / max(-aggregate_invest_pos, eps) )
    - beta1_capex, beta2_net_invest ~= OLS of ΔOtherNCA on [capex, aggregate_invest]

    where aggregate_invest is the same net acquisitions/investing feature used in
    training (net business purchase/sale + net investment purchase/sale + other
    investing changes).

    If required rows are missing or data is too short, we return zeros so the
    structural layer falls back to flat balances.
    """

    # Locate balance sheet rows
    gw_row = _find_row(bal, "Goodwill")
    onca_row = _find_row(bal, "Other Non Current Assets")
    if gw_row is None and onca_row is None:
        return {
            "premium_ratio_goodwill": 0.0,
            "beta1_capex": 0.0,
            "beta2_net_invest": 0.0,
        }

    # Aggregate investing feature from CF (same construction as in train_wmt)
    def cf_row(tokens: list[str]) -> np.ndarray:
        r = _find_row(cf, tokens)
        if r is None:
            return np.zeros(cf.shape[1], dtype=np.float32)
        vals = cf.loc[r].to_numpy(dtype=np.float32)
        return np.nan_to_num(vals, nan=0.0)

    nbps = cf_row("Net Business Purchase And Sale")
    ninv = cf_row("Net Investment Purchase And Sale")
    noth = cf_row("Net Other Investing Changes")
    aggregate_invest = nbps + ninv + noth  # shape [T]

    T = aggregate_invest.size
    if T < 2:
        return {
            "premium_ratio_goodwill": 0.0,
            "beta1_capex": 0.0,
            "beta2_net_invest": 0.0,
        }

    # 1) premium_ratio_goodwill from ΔGoodwill vs acquisitions (-aggregate_invest)
    premium_ratio_goodwill = 0.0
    if gw_row is not None:
        gw = bal.loc[gw_row].to_numpy(dtype=np.float32)
        if gw.size == T:
            d_gw = gw[1:] - gw[:-1]
            acq = np.maximum(-aggregate_invest[1:], 0.0)
            mask = acq > EPS
            if np.any(mask):
                ratios = d_gw[mask] / np.maximum(acq[mask], EPS)
                premium_ratio_goodwill = float(np.median(ratios))

    # 2) beta1_capex, beta2_net_invest from ΔOtherNCA ~ capex + aggregate_invest
    beta1_capex = 0.0
    beta2_net_invest = 0.0
    if onca_row is not None:
        onca = bal.loc[onca_row].to_numpy(dtype=np.float32)
        if onca.size == T:
            d_onca = onca[1:] - onca[:-1]
            # Capex row from CF (cash outflow, usually negative)
            capex_row = None
            for r in cf.index:
                if "capital expenditure" in r.lower():
                    capex_row = r
                    break
            if capex_row is not None:
                capex_cf = -cf.loc[capex_row].to_numpy(dtype=np.float32)  # make positive outflow
                if capex_cf.size == T:
                    X1 = capex_cf[1:]
                    X2 = aggregate_invest[1:]
                    y = d_onca
                    X = np.stack([X1, X2], axis=1)  # [T-1, 2]
                    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                    if np.count_nonzero(mask) >= 2:
                        Xm = X[mask]
                        ym = y[mask]
                        # Solve (X^T X) beta = X^T y with small ridge for stability
                        XtX = Xm.T @ Xm + 1e-8 * np.eye(2, dtype=np.float32)
                        Xty = Xm.T @ ym
                        try:
                            beta = np.linalg.solve(XtX, Xty)
                            beta1_capex = float(beta[0])
                            beta2_net_invest = float(beta[1])
                        except np.linalg.LinAlgError:
                            beta1_capex = 0.0
                            beta2_net_invest = 0.0

    return {
        "premium_ratio_goodwill": premium_ratio_goodwill,
        "beta1_capex": beta1_capex,
        "beta2_net_invest": beta2_net_invest,
    }


def estimate_oca_coefficients(bal: pd.DataFrame, fin: pd.DataFrame) -> Dict[str, float]:
    """Estimate simple coefficients linking ΔOther Current Assets to ΔSales and Opex.

    Very low-variance, back-of-envelope model:

    - omega_oca_sales  ~= median( ΔOCA / max(ΔSales_pos, eps) )
    - omega_oca_opex   ~= median( ΔOCA / max(ΔOpex_pos,  eps) )

    We compute both and keep them as separate knobs; the structural layer can
    combine them as: ΔOCA_t ≈ omega_oca_sales * ΔSales_t + omega_oca_opex * Opex_t.

    If rows are missing or the series are too short, we fall back to zeros so
    the engine keeps OCA flat.
    """

    oca_row = _find_row(bal, "Other Current Assets")
    sales_row = _find_row(fin, "Total Revenue")
    opex_row = _find_row(fin, "Operating Expense")

    if oca_row is None or sales_row is None or opex_row is None:
        return {"omega_oca_sales": 0.0, "omega_oca_opex": 0.0}

    oca = bal.loc[oca_row].to_numpy(dtype=np.float32)
    sales = fin.loc[sales_row].to_numpy(dtype=np.float32)

    opex_vals = fin.loc[opex_row].to_numpy(dtype=np.float32)

    T = min(oca.size, sales.size, opex_vals.size)
    if T < 2:
        return {"omega_oca_sales": 0.0, "omega_oca_opex": 0.0}

    # Truncate to common length; for Walmart data should be aligned already
    oca = oca[:T]
    sales = sales[:T]
    opex_vals = opex_vals[:T]

    d_oca = oca[1:] - oca[:-1]
    d_sales = sales[1:] - sales[:-1]
    d_sales_pos = np.maximum(d_sales, 0.0)
    opex_pos = np.maximum(opex_vals[1:], 0.0)

    # omega_oca_sales from periods with positive Δsales
    mask_sales = d_sales_pos > EPS
    if np.any(mask_sales):
        ratios_sales = d_oca[mask_sales] / np.maximum(d_sales_pos[mask_sales], EPS)
        omega_sales = float(np.median(ratios_sales))
    else:
        omega_sales = 0.0

    # omega_oca_opex from periods with positive opex
    mask_opex = opex_pos > EPS
    if np.any(mask_opex):
        ratios_opex = d_oca[mask_opex] / np.maximum(opex_pos[mask_opex], EPS)
        omega_opex = float(np.median(ratios_opex))
    else:
        omega_opex = 0.0

    return {"omega_oca_sales": omega_sales, "omega_oca_opex": omega_opex}


def estimate_oncl_coefficients(cf: pd.DataFrame) -> Dict[str, float]:
    """Estimate coefficients linking ΔOther Non-Current Liabilities to deferred tax & other non-cash items.

    We use a very coarse, low-variance linear model on deltas:

        ΔONCL_t ≈ psi1 * DeferredTax_t + psi2 * OtherNonCashItems_t.

    Implementation detail:
    - We map ONCL from the balance-sheet row "Other Non Current Liabilities" in training,
      but here we only need CF-side magnitudes to produce reasonable scales.
    - For now we keep it simple and estimate psi_oncl_deferred_tax and psi_oncl_other_nc
      as medians of per-period ratios, mirroring the OCA treatment.

    If rows are missing or insufficient length, we fall back to zeros so the engine
    keeps ONCL effectively flat aside from calibration adjustments.
    """

    # Deferred income tax CF line
    def_row = _find_row(cf, "Deferred Income Tax")
    # Other non-cash items CF line
    other_row = _find_row(cf, "Other Non Cash Items")

    if def_row is None and other_row is None:
        return {"psi_oncl_deferred_tax": 0.0, "psi_oncl_other_nc": 0.0}

    if def_row is not None:
        deferred = cf.loc[def_row].to_numpy(dtype=np.float32)
    else:
        deferred = np.zeros(cf.shape[1], dtype=np.float32)

    if other_row is not None:
        other_nc = cf.loc[other_row].to_numpy(dtype=np.float32)
    else:
        other_nc = np.zeros(cf.shape[1], dtype=np.float32)

    T = min(deferred.size, other_nc.size)
    if T == 0:
        return {"psi_oncl_deferred_tax": 0.0, "psi_oncl_other_nc": 0.0}

    # Very crude scale heuristics: how much ONCL moves per unit of each flow.
    # We don't observe ΔONCL here (that's from BS); instead, we just set
    # coefficients so that unit deferred tax / other non-cash roughly pass
    # through to ONCL (order-of-magnitude 1). Calibration will refine.
    # To avoid crazy magnitudes, bound by small absolute values.
    # Use medians to avoid outliers.
    def_mag = np.median(np.abs(deferred)) if np.any(deferred) else 0.0
    other_mag = np.median(np.abs(other_nc)) if np.any(other_nc) else 0.0

    # If a series is essentially zero, use zero coefficient; otherwise start
    # from a modest pass-through (≈ 0.1) so that ONCL reacts, but weakly.
    psi_def = 0.0 if def_mag == 0.0 else 0.1
    psi_other = 0.0 if other_mag == 0.0 else 0.1

    return {"psi_oncl_deferred_tax": psi_def, "psi_oncl_other_nc": psi_other}


def build_policies_from_dict(d: Dict[str, tf.Tensor]) -> PoliciesWMT:
    """Helper to construct PoliciesWMT from a flat dict of tensors.

    Any optional coefficients not provided will default to None/zeros in the engine.
    """
    return PoliciesWMT(
        inflation=d["inflation"],
        real_st_rate=d["real_st_rate"],
        real_lt_rate=d["real_lt_rate"],
        tax_rate=d["tax_rate"],
        payout_ratio=d["payout_ratio"],
        min_cash_ratio=d["min_cash_ratio"],
        lt_share_for_capex=d["lt_share_for_capex"],
        st_invest_spread=d["st_invest_spread"],
        debt_spread=d["debt_spread"],
        dso_days=d["dso_days"],
        dpo_days=d["dpo_days"],
        dio_days=d["dio_days"],
        opex_ratio=d["opex_ratio"],
        depreciation_rate=d["depreciation_rate"],
        gross_margin=d["gross_margin"],
        # New: constant paths for investing sensitivities so that the structural
        # layer can move goodwill and other NCA in both calibration and forecast.
        premium_ratio_goodwill=d.get("premium_ratio_goodwill"),
        beta1_capex=d.get("beta1_capex"),
        beta2_net_invest=d.get("beta2_net_invest"),
        # Lease schedule defaults: simple, constant, and currently not trained.
        lease_addition_capex_coeff=d.get("lease_addition_capex_coeff"),
        lease_addition_sales_coeff=d.get("lease_addition_sales_coeff"),
        lease_avg_remaining_term=d.get("lease_avg_remaining_term"),
        lease_principal_payment_rate=d.get("lease_principal_payment_rate"),
        # Other current assets fallback coefficients
        omega_oca_sales=d.get("omega_oca_sales"),
        omega_oca_opex=d.get("omega_oca_opex"),
        # Other non-current liabilities fallback coefficients
        psi_oncl_deferred_tax=d.get("psi_oncl_deferred_tax"),
        psi_oncl_other_nc=d.get("psi_oncl_other_nc"),
        gamma_capital_stock=d.get("gamma_capital_stock"),
        k_pi=d.get("k_pi"),
    )


def main(horizon_quarters: int = 1) -> None:
    data = load_wmt_quarterlies()
    bal_all = data["bal"]
    fin_all = data["fin"]
    cf_all = data["cf"]

    # Explicitly enforce train/validation split in time:
    # training quarters: 2024-07-31, 2024-10-31, 2025-01-31, 2025-04-30
    # validation quarter (held out from estimation): 2025-07-31
    train_cols = [
        "2024-07-31",
        "2024-10-31",
        "2025-01-31",
        "2025-04-30",
    ]
    # Filter to whichever of these are actually present, but require at least 2
    # quarters to avoid degenerate estimates.
    train_cols = [c for c in train_cols if c in fin_all.columns]
    if len(train_cols) < 2:
        raise ValueError(
            f"Not enough training quarters available. Expected at least 2 of {train_cols}, "
            f"found columns {list(fin_all.columns)}"
        )

    bal = bal_all[train_cols]
    fin = fin_all[train_cols]
    cf = cf_all[train_cols]

    # Use only training quarters to estimate levels, then extend as constant
    # for forecast horizon.
    pol_dict = learn_basic_policies(bal, fin, cf)
    T_hist = fin.shape[1]

    # Extend each policy tensor forward by holding last value constant
    def extend_const(t: tf.Tensor, h: int) -> tf.Tensor:
        # t shape: [1,T_hist,1]
        last = t[:, -1:, :]
        ext = tf.repeat(last, h, axis=1)
        return tf.concat([t, ext], axis=1)

    pol_dict_fwd: Dict[str, tf.Tensor] = {}
    for k, v in pol_dict.items():
        pol_dict_fwd[k] = extend_const(v, horizon_quarters)

    # Estimate sensitivities before building PoliciesWMT so we can inject
    # them as constant paths.
    inv_sens = estimate_investing_sensitivity(bal, cf)
    # Estimate simple Other Current Assets coefficients.
    oca_coeffs = estimate_oca_coefficients(bal, fin)
    # Estimate coarse Other Non-Current Liabilities coefficients.
    oncl_coeffs = estimate_oncl_coefficients(cf)

    # Add constant tensors for the new coefficients into the policy dict.
    total_T = T_hist + horizon_quarters
    pol_dict_fwd["premium_ratio_goodwill"] = constant_tensor(
        inv_sens["premium_ratio_goodwill"], total_T
    )
    pol_dict_fwd["beta1_capex"] = constant_tensor(
        inv_sens["beta1_capex"], total_T
    )
    pol_dict_fwd["beta2_net_invest"] = constant_tensor(
        inv_sens["beta2_net_invest"], total_T
    )

    # Add constant tensors for OCA coefficients.
    pol_dict_fwd["omega_oca_sales"] = constant_tensor(
        oca_coeffs["omega_oca_sales"], total_T
    )
    pol_dict_fwd["omega_oca_opex"] = constant_tensor(
        oca_coeffs["omega_oca_opex"], total_T
    )

    # Add constant tensors for ONCL coefficients.
    pol_dict_fwd["psi_oncl_deferred_tax"] = constant_tensor(
        oncl_coeffs["psi_oncl_deferred_tax"], total_T
    )
    pol_dict_fwd["psi_oncl_other_nc"] = constant_tensor(
        oncl_coeffs["psi_oncl_other_nc"], total_T
    )

    # --- Lease schedule constant defaults ---
    # These are deliberately simple and not yet estimated from data.
    # - New lease additions scale weakly with capex and sales growth.
    # - Average remaining term is set to 20 quarters (~5 years).
    # - Principal payment rate approximates 1 / average term.
    lease_add_capex_coeff = 0.02  # small fraction of capex treated as new leases
    lease_add_sales_coeff = 0.01   # start with no direct dependence on sales growth
    lease_avg_term_q = 20.0       # quarters
    lease_principal_rate = 1.0 / lease_avg_term_q

    pol_dict_fwd["lease_addition_capex_coeff"] = constant_tensor(
        lease_add_capex_coeff, total_T
    )
    pol_dict_fwd["lease_addition_sales_coeff"] = constant_tensor(
        lease_add_sales_coeff, total_T
    )
    pol_dict_fwd["lease_avg_remaining_term"] = constant_tensor(
        lease_avg_term_q, total_T
    )
    pol_dict_fwd["lease_principal_payment_rate"] = constant_tensor(
        lease_principal_rate, total_T
    )

    gamma_capital_stock = 1.0  # modest growth in capital stock by default
    pol_dict_fwd["gamma_capital_stock"] = constant_tensor(
        gamma_capital_stock, total_T
    )
    k_pi = -8.0  # neutral price-capital elasticity by default
    pol_dict_fwd["k_pi"] = constant_tensor(
        k_pi, total_T
    )

    policies_fwd = build_policies_from_dict(pol_dict_fwd)

    # Estimate a simple capex growth coefficient for driver simulation
    gamma_capex_growth = estimate_capex_growth_coeff(fin, cf)
    # Estimate a simple AOCI drift parameter for forward change_in_aoci
    aoci_drift = estimate_aoci_drift(bal)

    # Persist learned artifacts as simple numpy arrays for now
    def tensor_to_np(x: tf.Tensor) -> np.ndarray:
        return x.numpy()

    np.savez(OUT_DIR / "wmt_policies_forward.npz",
             inflation=tensor_to_np(policies_fwd.inflation),
             real_st_rate=tensor_to_np(policies_fwd.real_st_rate),
             real_lt_rate=tensor_to_np(policies_fwd.real_lt_rate),
             tax_rate=tensor_to_np(policies_fwd.tax_rate),
             payout_ratio=tensor_to_np(policies_fwd.payout_ratio),
             min_cash_ratio=tensor_to_np(policies_fwd.min_cash_ratio),
             lt_share_for_capex=tensor_to_np(policies_fwd.lt_share_for_capex),
             st_invest_spread=tensor_to_np(policies_fwd.st_invest_spread),
             debt_spread=tensor_to_np(policies_fwd.debt_spread),
             dso_days=tensor_to_np(policies_fwd.dso_days),
             dpo_days=tensor_to_np(policies_fwd.dpo_days),
             dio_days=tensor_to_np(policies_fwd.dio_days),
             opex_ratio=tensor_to_np(policies_fwd.opex_ratio),
             depreciation_rate=tensor_to_np(policies_fwd.depreciation_rate),
             gross_margin=tensor_to_np(policies_fwd.gross_margin),
             premium_ratio_goodwill=tensor_to_np(policies_fwd.premium_ratio_goodwill),
             beta1_capex=tensor_to_np(policies_fwd.beta1_capex),
             beta2_net_invest=tensor_to_np(policies_fwd.beta2_net_invest),
             omega_oca_sales=tensor_to_np(policies_fwd.omega_oca_sales),
             omega_oca_opex=tensor_to_np(policies_fwd.omega_oca_opex),
             psi_oncl_deferred_tax=tensor_to_np(policies_fwd.psi_oncl_deferred_tax),
             psi_oncl_other_nc=tensor_to_np(policies_fwd.psi_oncl_other_nc),
             lease_addition_capex_coeff=tensor_to_np(policies_fwd.lease_addition_capex_coeff),
             lease_addition_sales_coeff=tensor_to_np(policies_fwd.lease_addition_sales_coeff),
             lease_avg_remaining_term=tensor_to_np(policies_fwd.lease_avg_remaining_term),
             lease_principal_payment_rate=tensor_to_np(policies_fwd.lease_principal_payment_rate),
             gamma_capital_stock=tensor_to_np(policies_fwd.gamma_capital_stock),
             k_pi=tensor_to_np(policies_fwd.k_pi),
             )

    np.savez(OUT_DIR / "wmt_driver_params.npz",
             gamma_capex_growth=gamma_capex_growth,
             aoci_drift=aoci_drift,
             premium_ratio_goodwill=inv_sens["premium_ratio_goodwill"],
             beta1_capex=inv_sens["beta1_capex"],
             beta2_net_invest=inv_sens["beta2_net_invest"],
             )

    print("Saved learned WMT policies and driver params to", OUT_DIR)
    opr, op, s = pseudo_learn_opex_ratio(fin)
    print(opr, op, s)


if __name__ == "__main__":
    main()
