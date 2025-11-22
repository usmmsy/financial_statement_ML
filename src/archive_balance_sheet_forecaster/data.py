from __future__ import annotations
try:
    import yfinance as yf
except Exception:  # optional dependency
    yf = None
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os

from balance_sheet_forecaster.types import Policies, PrevState


# Extract data from Yahoo's nested dictionaries

def _extract_quarterly_dict(
        yf_block: Dict,
        ticker: str,
) -> Dict[str, Dict]:
    """
    yf_block format:
    {
        'balanceSheetHistoryQuarterly': {
            'MSFT': [
                { ... },
                { ... },
                ...
            ],
            'AAPL': [
                { ... },
                { ... },
                ...
            ]
        }
    }

    For the given ticker, returns a dictionary mapping date strings to data dictionaries:
    {
        '2023-06-30': { ... },
        '2023-03-31': { ... },
        ...
    }

    If key not found, returns empty dict.
    """

    if not isinstance(yf_block, dict):
        return {}
    
    preferred = [
        "incomeStatementHistoryQuarterly",
        "balanceSheetHistoryQuarterly",
        "cashflowStatementHistoryQuarterly",
    ]
    
    top_key = next((k for k in preferred if k in yf_block), next(iter(yf_block.keys()), None))
    per_ticker_list = yf_block.get(top_key, {}).get(ticker, [])
    out = {}

    for entry in per_ticker_list:
        # entry format: { '2023-06-30': { ... }, ... }
        if not isinstance(entry, dict):
            continue

        for period, values in entry.items():
            out[period] = values

    return out

def _quarterly_dict_to_df(
        quarterly_dict: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Converts a dictionary mapping date strings to data dictionaries into a DataFrame.
    Rows are indexed by date strings, columns are the keys from the data dictionaries.
    Missing values are filled with NaN.
    """

    if not quarterly_dict:
        return pd.DataFrame()

    rows = []
    for period, values in quarterly_dict.items():
        try:
            date = pd.to_datetime(period)
        except Exception:
            date = period  # Keep as string if conversion fails
        row = {'date': date}
        if isinstance(values, dict):
            for key, value in values.items():
                row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)

    return df


def _pick_first_key(
        d: Dict,
        keys: List[str],
        default=None,
):
    # Returns the value for the first key found in d from the keys list.
    # Helper for handling multiple possible key names.
    for key in keys:
        if key in d:
            return d[key]
    return default


# Feature engineering

# ---------------- Optional lightweight preprocessing knobs ----------------
# These globals allow plug-in pre-fit preprocessing parameters
# (kept off by default so existing behavior and tests remain unchanged).

# Per-feature winsorization thresholds for days features, e.g.:
# {
#   "dso_days": (lo, hi),
#   "dpo_days": (lo, hi),
#   "dio_days": (lo, hi),
# }
_DAYS_WINSOR_THRESHOLDS: Dict[str, Tuple[float | None, float | None]] | None = None

# Whether to apply log1p to days features after winsorization
_USE_LOG1P_FOR_DAYS: bool = False

# Optional feature scaler: either
# - a dict with arrays: {"means": np.ndarray[F], "stds": np.ndarray[F]}
# - or a dict with per-feature stats keyed by name: {name: {"mean": float, "std": float}}
# The feature order is defined by _FEATURE_NAMES below.
_FEATURE_SCALER: Dict | None = None

# Feature order used in _build_features_panel
_FEATURE_NAMES: Tuple[str, ...] = (
    "sales",
    "revenue_growth",
    "gross_margin",
    "opex_ratio",
    "debt_to_equity",
    "dso_days",
    "dpo_days",
    "dio_days",
    "capex_intensity",
    "payout_ratio",
)


def set_feature_preprocessing(
    days_winsor_thresholds: Dict[str, Tuple[float | None, float | None]] | None = None,
    use_log1p_for_days: bool = False,
    feature_scaler: Dict | None = None,
) -> None:
    """
    Configure optional preprocessing applied in _build_features_panel.

    Parameters
    - days_winsor_thresholds: dict mapping {dso_days|dpo_days|dio_days -> (lo, hi)};
      lo/hi can be None to skip that bound. If None, no winsorization.
    - use_log1p_for_days: if True, apply np.log1p to the (optionally winsorized) days features.
    - feature_scaler: per-feature scaler stats. Either arrays {"means": [F], "stds": [F]},
      or a dict keyed by feature name with {"mean", "std"} entries. If None, no scaling.

    These settings are global and affect subsequent calls to _build_features_panel.
    """
    global _DAYS_WINSOR_THRESHOLDS, _USE_LOG1P_FOR_DAYS, _FEATURE_SCALER
    _DAYS_WINSOR_THRESHOLDS = days_winsor_thresholds
    _USE_LOG1P_FOR_DAYS = bool(use_log1p_for_days)
    _FEATURE_SCALER = feature_scaler


def _winsorize(x: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    if lo is None and hi is None:
        return x
    lo_v = -np.inf if lo is None else float(lo)
    hi_v = np.inf if hi is None else float(hi)
    return np.clip(x, lo_v, hi_v)


def _log1p_if(flag: bool, x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0)) if flag else x


def _apply_feature_scaler(feats: np.ndarray, scaler: Dict | None) -> np.ndarray:
    """
    Apply per-feature scaling using either arrays or name-keyed stats.
    Expects feats shape [T, F] with feature order = _FEATURE_NAMES.
    """
    if scaler is None:
        return feats
    eps = 1e-8
    if "means" in scaler and "stds" in scaler:
        means = np.asarray(scaler["means"], dtype=np.float32)
        stds = np.asarray(scaler["stds"], dtype=np.float32)
        if means.shape[-1] != feats.shape[-1] or stds.shape[-1] != feats.shape[-1]:
            raise ValueError("FEATURE_SCALER arrays must match feature dimension F")
        return (feats - means) / (np.maximum(stds, eps))
    # name-keyed form
    means = np.zeros((feats.shape[-1],), dtype=np.float32)
    stds = np.ones((feats.shape[-1],), dtype=np.float32)
    for i, name in enumerate(_FEATURE_NAMES):
        stats = scaler.get(name)
        if stats is None:
            continue
        m = float(stats.get("mean", 0.0))
        s = float(stats.get("std", 1.0))
        means[i] = m
        stds[i] = s
    return (feats - means) / (np.maximum(stds, eps))

def _safe_ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
        floor: float = 1e-6,
) -> np.ndarray:
    return np.divide(
        numerator, 
        np.maximum(np.abs(denominator), floor)
    )


def _safe_ratio_signed(
        numerator: np.ndarray,
        denominator: np.ndarray,
        floor: float = 1e-6,
) -> np.ndarray:
    """
    Sign-preserving safe division.

    Unlike _safe_ratio (which uses |denominator|), this keeps the sign of the
    denominator by applying the floor with sign: max(|den|, floor) * sign(den).

    This is useful for economics/accounting ratios where the sign of the base
    (e.g., equity, net income, sales) conveys distress or regime changes.
    """
    den = denominator.astype(np.float32)
    mag = np.maximum(np.abs(den), floor)
    signed_den = np.where(den >= 0.0, mag, -mag)
    return np.divide(numerator, signed_den)


def _days_on_hand(
        balance: np.ndarray,
        flow: np.ndarray,
) -> np.ndarray:
    # balance ~= flow * days / 365 => days ~= balance / flow * 365
    return _safe_ratio(balance, flow) * 365

def _build_features_panel(
        df_inc: pd.DataFrame,
        df_bs: pd.DataFrame,
        df_cf: pd.DataFrame,
) -> pd.DataFrame:
    
    """
    Build a [T, F] numpy array of engineered features for DriverHead,
    using only observable historical quantities.

    We'll include:
        1. revenue growth
        2. gross profit margin
        3. opex ratio
        4. leverage ratios
        5. DSO/DPO/DSI
        6. capex intensity
        7. payout ratio (dividends / net income)

    Note: Can add macro inputs later.
    """

    # Income statement features
    sales_arr = df_inc["sales"].to_numpy(dtype=np.float32)
    cogs_arr = df_inc["cogs"].to_numpy(dtype=np.float32)
    opex_arr = df_inc["opex"].to_numpy(dtype=np.float32)
    net_income_arr = df_inc["net_income"].to_numpy(dtype=np.float32)

    # Balance sheet features
    ar_arr = df_bs["ar"].to_numpy(dtype=np.float32)
    ap_arr = df_bs["ap"].to_numpy(dtype=np.float32)
    inv_arr = df_bs["inventory"].to_numpy(dtype=np.float32)
    st_debt_arr = df_bs["st_debt"].to_numpy(dtype=np.float32)
    lt_debt_arr = df_bs["lt_debt"].to_numpy(dtype=np.float32)
    equity_arr = df_bs["equity"].to_numpy(dtype=np.float32)

    # Cash flow statement features
    capex_arr = df_cf["capex"].to_numpy(dtype=np.float32)
    dividends_arr = df_cf["dividends"].to_numpy(dtype=np.float32)

    # Derived features
    revenue_growth = np.zeros_like(sales_arr)
    revenue_growth[1:] = _safe_ratio(
        sales_arr[1:] - sales_arr[:-1],
        np.abs(sales_arr[:-1])
    )

    # Gross profit margin = 1 - COGS / Sales (preserve sign of Sales)
    gross_margin = 1.0 - _safe_ratio_signed(cogs_arr, sales_arr)

    # Opex ratio = Opex / Sales (preserve sign of Sales)
    opex_ratio = _safe_ratio_signed(opex_arr, sales_arr)

    # Leverage ratios
    debt_total = st_debt_arr + lt_debt_arr
    # Leverage: preserve sign of equity so negative equity is visible
    debt_to_equity = _safe_ratio_signed(debt_total, equity_arr)

    # Working capital timing: realized DSO, DPO, DIO (raw days)
    dso_days = _days_on_hand(ar_arr, sales_arr)
    dpo_days = _days_on_hand(ap_arr, cogs_arr)
    dio_days = _days_on_hand(inv_arr, cogs_arr)

    # Optional: apply pre-set winsorization/log1p to days features,
    # configured globally via set_feature_preprocessing(). Defaults: no-op.
    if _DAYS_WINSOR_THRESHOLDS is not None:
        dso_lo, dso_hi = _DAYS_WINSOR_THRESHOLDS.get("dso_days", (None, None))
        dpo_lo, dpo_hi = _DAYS_WINSOR_THRESHOLDS.get("dpo_days", (None, None))
        dio_lo, dio_hi = _DAYS_WINSOR_THRESHOLDS.get("dio_days", (None, None))
        dso_days = _winsorize(dso_days, dso_lo, dso_hi)
        dpo_days = _winsorize(dpo_days, dpo_lo, dpo_hi)
        dio_days = _winsorize(dio_days, dio_lo, dio_hi)

    if _USE_LOG1P_FOR_DAYS:
        dso_days = _log1p_if(True, dso_days)
        dpo_days = _log1p_if(True, dpo_days)
        dio_days = _log1p_if(True, dio_days)

    # Capex intensity = Capex / Sales (flip sign since capex is negative cash flow)
    capex_intensity = _safe_ratio(-capex_arr, sales_arr)

    # Payout ratio = Dividends / Net Income (flip sign since dividends are negative cash flow)
    # Preserve sign of net income to reflect unsustainable payouts when NI <= 0
    payout_ratio = _safe_ratio_signed(-dividends_arr, net_income_arr)

    # Stack features [T, F]
    feats = np.stack(
        [
            sales_arr,
            revenue_growth,
            gross_margin,
            opex_ratio,
            debt_to_equity,
            dso_days,
            dpo_days,
            dio_days,
            capex_intensity,
            payout_ratio,
        ],
        axis=-1,
    ).astype(np.float32)

    # Optional: apply per-feature scaling with pre-set stats
    feats = _apply_feature_scaler(feats, _FEATURE_SCALER)

    return feats


# Converting per-ticker DataFrames -> model tensors

def _df_to_targets(
        df_inc: pd.DataFrame,
        df_bs: pd.DataFrame,
) -> np.ndarray:
    
    """
    Produce ground-truth statement lines the model should match.
    Each output is [T, 1]
    """

    def col(df, name):
        return df[name].to_numpy(dtype=np.float32).reshape(-1, 1)
    
    targets = {
        "sales": col(df_inc, "sales") if "sales" in df_inc.columns else np.zeros((len(df_inc), 1), dtype=np.float32),
        "cogs": col(df_inc, "cogs") if "cogs" in df_inc.columns else np.zeros((len(df_inc), 1), dtype=np.float32),
        "opex": col(df_inc, "opex") if "opex" in df_inc.columns else np.zeros((len(df_inc), 1), dtype=np.float32),
        "net_income": col(df_inc, "net_income") if "net_income" in df_inc.columns else np.zeros((len(df_inc), 1), dtype=np.float32),
        # Optional IS lines for richer supervision
        "interest": col(df_inc, "interest_expense") if "interest_expense" in df_inc.columns else np.zeros((len(df_inc), 1), dtype=np.float32),
        "tax": col(df_inc, "income_tax_expense") if "income_tax_expense" in df_inc.columns else np.zeros((len(df_inc), 1), dtype=np.float32),
        "cash": col(df_bs, "cash") if "cash" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "st_investments": col(df_bs, "st_investments") if "st_investments" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "ar": col(df_bs, "ar") if "ar" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "ap": col(df_bs, "ap") if "ap" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "inventory": col(df_bs, "inventory") if "inventory" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "st_debt": col(df_bs, "st_debt") if "st_debt" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "lt_debt": col(df_bs, "lt_debt") if "lt_debt" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "nfa": col(df_bs, "nfa") if "nfa" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
        "equity": col(df_bs, "equity") if "equity" in df_bs.columns else np.zeros((len(df_bs), 1), dtype=np.float32),
    }

    return targets


def _df_to_prevstate(df_bs: pd.DataFrame) -> Dict[str, float]:
    """
    Extract previous balance sheet state from the last row of df_bs.
    """

    last_row = df_bs.iloc[-1]

    prev_state = {
        "cash": float(last_row["cash"]),
        "st_investments": float(last_row["st_investments"]),
        "st_debt": float(last_row["st_debt"]),
        "lt_debt": float(last_row["lt_debt"]),
        "ar": float(last_row["ar"]),
        "ap": float(last_row["ap"]),
        "inventory": float(last_row["inventory"]),
        "nfa": float(last_row["nfa"]),
        "equity": float(last_row["equity"]),
    }

    return prev_state


# Per-ticker normalization


def _normalize_ticker_frames(
    ticker: str,
    inc_q: Dict,
    bs_q: Dict,
    cf_q: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Build aligned quarterly DataFrames for one ticker with canonical columns.
    """

    q_inc = _extract_quarterly_dict(inc_q, ticker)
    q_bs  = _extract_quarterly_dict(bs_q, ticker)
    q_cf  = _extract_quarterly_dict(cf_q, ticker)

    df_inc_raw = _quarterly_dict_to_df(q_inc)
    df_bs_raw  = _quarterly_dict_to_df(q_bs)
    df_cf_raw  = _quarterly_dict_to_df(q_cf)

    # Map Yahoo keys to canonical model keys for income
    def map_inc_row(r: pd.Series) -> Dict[str, float]:
        d = r.to_dict()
        sales = _pick_first_key(d, ["totalRevenue", "Total Revenue", "total_revenue"])
        cogs  = _pick_first_key(d, ["costOfRevenue", "Cost Of Revenue", "cost_of_revenue"])
        sgna  = _pick_first_key(d, ["sellingGeneralAdministrative", "sellingGeneralAndAdministrative", "Selling General Administrative"])
        rnd   = _pick_first_key(d, ["researchDevelopment", "researchAndDevelopment", "Research Development"])
        neti  = _pick_first_key(d, ["netIncome", "Net Income", "net_income"])
        # Interest and tax (various Yahoo key variants)
        interest_exp = _pick_first_key(
            d,
            [
                "interestExpense",
                "interestExpenseNonOperating",
                "Interest Expense",
                "interest_expense",
            ],
        )
        income_tax_exp = _pick_first_key(
            d,
            [
                "incomeTaxExpense",
                "incomeTaxProvision",
                "Income Tax Expense",
                "income_tax_expense",
            ],
        )
        # Optional: interest income should the needs come later
        interest_inc = _pick_first_key(
            d,
            [
                "interestIncome",
                "nonOperatingIncomeInterestIncome",
                "Interest Income",
                "interest_income",
            ],
        )

        return {
            "sales": sales,
            "cogs": cogs,
            "opex": (sgna or 0.0) + (rnd or 0.0),
            "net_income": neti,
            # Keep separate for clarity; training can choose to use these or not
            "interest_expense": interest_exp,
            "income_tax_expense": income_tax_exp,
            "interest_income": interest_inc,
        }

    def map_bs_row(r: pd.Series) -> Dict[str, float]:
        d = r.to_dict()
        return {
            "cash": _pick_first_key(d, ["cashAndCashEquivalents", "cashAndCashEquivalentsAtCarryingValue", "cash"]),
            "st_investments": _pick_first_key(d, ["shortTermInvestments", "shortTermInvestmentsOther"]),
            "ar": _pick_first_key(d, ["netReceivables", "accountsReceivable", "accountsReceivableNetCurrent"]),
            "ap": _pick_first_key(d, ["accountsPayable", "accountsPayableCurrent"]),
            "inventory": _pick_first_key(d, ["inventory", "inventoryNet", "inventories"]),
            "st_debt": _pick_first_key(d, ["shortLongTermDebt", "shortTermDebt", "currentPortionOfLongTermDebt"]),
            "lt_debt": _pick_first_key(d, ["longTermDebt", "longTermDebtNoncurrent"]),
            "equity": _pick_first_key(d, ["totalStockholderEquity", "totalStockholdersEquity", "stockholdersEquity"]),
            "nfa": _pick_first_key(d, ["propertyPlantEquipmentNet", "propertyPlantEquipment", "netPPE"]),
            # Optional tax-related balances for future use
            "tax_payable": _pick_first_key(
                d,
                [
                    "incomeTaxesPayable",
                    "currentTaxLiabilities",
                    "incomeTaxPayable",
                ],
            ),
            "deferred_tax_liabilities": _pick_first_key(
                d,
                [
                    "deferredTaxLiabilities",
                    "deferredTaxLiabilitiesNoncurrent",
                    "deferredIncomeTaxLiabilities",
                ],
            ),
            "deferred_tax_assets": _pick_first_key(
                d,
                [
                    "deferredTaxAssets",
                    "deferredTaxAssetsNoncurrent",
                ],
            ),
        }

    def map_cf_row(r: pd.Series) -> Dict[str, float]:
        d = r.to_dict()
        capex = _pick_first_key(d, ["capitalExpenditures", "capitalExpenditure"])
        divs  = _pick_first_key(d, ["dividendsPaid", "cashDividendsPaid"])
        return {
            "capex": float(capex) if capex is not None else 0.0,
            "dividends": float(divs) if divs is not None else 0.0,
        }

    df_inc = pd.DataFrame([map_inc_row(r) for _, r in df_inc_raw.iterrows()], index=df_inc_raw.index)
    df_bs  = pd.DataFrame([map_bs_row(r)  for _, r in df_bs_raw.iterrows()],  index=df_bs_raw.index)
    df_cf  = pd.DataFrame([map_cf_row(r)  for _, r in df_cf_raw.iterrows()],  index=df_cf_raw.index)

    # Align on common quarters (inner join on index)
    idx = df_inc.index.intersection(df_bs.index).intersection(df_cf.index)
    df_inc = df_inc.loc[idx].sort_index()
    df_bs  = df_bs.loc[idx].sort_index()
    df_cf  = df_cf.loc[idx].sort_index()

    return df_inc, df_bs, df_cf


def _build_per_ticker(ticker: str,
                      inc_q: Dict,
                      bs_q: Dict,
                      cf_q: Dict,
                      horizon_quarters: int) -> Dict[str, object] | None:
    df_inc, df_bs, df_cf = _normalize_ticker_frames(ticker, inc_q, bs_q, cf_q)

    if len(df_inc) < horizon_quarters or len(df_bs) < horizon_quarters or len(df_cf) < horizon_quarters:
        return None

    # slice most recent horizon_quarters
    df_inc = df_inc.iloc[-horizon_quarters:]
    df_bs  = df_bs.iloc[-horizon_quarters:]
    df_cf  = df_cf.iloc[-horizon_quarters:]

    features_np = _build_features_panel(df_inc, df_bs, df_cf)  # [T,F]
    targets_np  = _df_to_targets(df_inc, df_bs)                 # dict -> [T,1]
    prev_dict   = _df_to_prevstate(df_bs)                       # last-quarter snapshot

    return {
        "features": features_np,
        "targets": targets_np,
        "prev": prev_dict,
        "T": len(df_inc),
    }


# Batch stack and public loader


class YahooFinancialsLoader:
    """
    DEPRECATED shim retained for tests that call _stack_batch without network.

    Use YFinanceLoader for real data fetching. This class no longer fetches data
    from the network; its _fetch_raw/all methods will raise at runtime.
    """

    def __init__(self,
                 tickers: List[str],
                 horizon_quarters: int = 8,
                 min_cash_ratio_default: float = 0.05,
                 payout_ratio_default: float = 0.20,
                 tax_rate_default: float = 0.25,
                 inflation_default: float = 0.02,
                 real_rate_default: float = 0.01):
        self.tickers = tickers
        self.horizon = horizon_quarters
        self.defaults = dict(
            min_cash_ratio=min_cash_ratio_default,
            payout_ratio=payout_ratio_default,
            tax_rate=tax_rate_default,
            inflation=inflation_default,
            real_rate=real_rate_default,
        )

    def _fetch_raw(self):
        raise RuntimeError("YahooFinancialsLoader is deprecated. Use YFinanceLoader instead.")
    
    def _stack_batch(
            self,
            per_ticker_data: Dict[str, Dict[str, object]],
    ) -> Tuple[tf.Tensor, Policies, PrevState, Dict[str, tf.Tensor], List[str]]:
        
        tickers_kept = list(per_ticker_data.keys())
        if not tickers_kept:
            raise ValueError("No valid tickers with sufficient data found.")
        
        # Stack features [B, T, F]
        feats_list = [per_ticker_data[t]["features"] for t in tickers_kept]
        feats_np = np.stack(feats_list, axis=0)
        B, T, F = feats_np.shape
        feats_tf = tf.convert_to_tensor(feats_np, dtype=tf.float32)

        # Stack targets dict -> [B, T, 1]
        all_keys = set()
        for t in tickers_kept:
            all_keys |= set(per_ticker_data[t]["targets"].keys())

        targets_tf: Dict[str, tf.Tensor] = {}
        for key in all_keys:
            rows = []
            for t in tickers_kept:
                arr = per_ticker_data[t]["targets"].get(key)
                if arr is None:
                    arr = np.zeros((T, 1), dtype=np.float32)
                rows.append(arr)
            stacked = np.stack(rows, axis=0)
            targets_tf[key] = tf.convert_to_tensor(stacked, dtype=tf.float32)

        # Prevstate [B, 1] for each field
        def stack_prev_field(field: str) -> tf.Tensor:
            vals = [[per_ticker_data[t]["prev"][field]] for t in tickers_kept]
            return tf.convert_to_tensor(np.array(vals, dtype=np.float32), dtype=tf.float32)
        
        prev_state = PrevState(
            cash=stack_prev_field("cash"),
            st_investments=stack_prev_field("st_investments"),
            st_debt=stack_prev_field("st_debt"),
            lt_debt=stack_prev_field("lt_debt"),
            ar=stack_prev_field("ar"),
            ap=stack_prev_field("ap"),
            inventory=stack_prev_field("inventory"),
            nfa=stack_prev_field("nfa"),
            equity=stack_prev_field("equity"),
        )

        def broadcast(val: float) -> tf.Tensor:
            return tf.ones((B, T, 1), dtype=tf.float32) * float(val)
        
        policies = Policies(
            min_cash_ratio=broadcast(self.defaults["min_cash_ratio"]),
            payout_ratio=broadcast(self.defaults["payout_ratio"]),
            tax_rate=broadcast(self.defaults["tax_rate"]),
            inflation=broadcast(self.defaults["inflation"]),
            real_rate=broadcast(self.defaults["real_rate"]),
        )

        return feats_tf, policies, prev_state, targets_tf, tickers_kept
    

    def all(self) -> Tuple[tf.Tensor, Policies, PrevState, Dict[str, tf.Tensor], List[str]]:
        """
        Fetch, process, and return all data as model-ready tensors.
        """

        inc_q, bs_q, cf_q = self._fetch_raw()

        per_ticker: Dict[str, Dict[str, object]] = {}
        for ticker in self.tickers:
            built = _build_per_ticker(
                ticker,
                inc_q,
                bs_q,
                cf_q,
                self.horizon,
            )
            if built is not None and built["T"] >= self.horizon:
                per_ticker[ticker] = built

        return self._stack_batch(per_ticker)


class YFinanceLoader:
    """
    Loader that pulls quarterly financials using yfinance.Ticker APIs.

    Produces the same outputs as YahooFinancialsLoader:
        features: [B, T, F]
        policies: Policies with each field [B, T, 1]
        prev: PrevState with each field [B, 1]
        targets: Dict[str->[B, T, 1]]
        tickers_used: List[str]

    Notes:
        - Requires yfinance; if unavailable, raise ImportError.
        - Some tickers may lack sufficient quarterly history; they are skipped.
    """

    def __init__(
        self,
        tickers: List[str],
        horizon_quarters: int = 8,
        min_cash_ratio_default: float = 0.05,
        payout_ratio_default: float = 0.20,
        tax_rate_default: float = 0.25,
        inflation_default: float = 0.02,
        real_rate_default: float = 0.01,
        per_request_delay_sec: float = 2.0,
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
    ):
        if yf is None:
            raise ImportError("yfinance is required for YFinanceLoader but is not installed.")
        self.tickers = tickers
        self.horizon = horizon_quarters
        self.defaults = dict(
            min_cash_ratio=min_cash_ratio_default,
            payout_ratio=payout_ratio_default,
            tax_rate=tax_rate_default,
            inflation=inflation_default,
            real_rate=real_rate_default,
        )
        # Simple throttling to mitigate Yahoo/yfinance rate limits
        self.per_request_delay_sec = float(
            os.getenv("YF_REQUEST_DELAY_SEC", per_request_delay_sec)
        )
        self.max_retries = int(
            os.getenv("YF_MAX_RETRIES", max_retries)
        )
        self.retry_backoff_base_sec = float(
            os.getenv("YF_BACKOFF_BASE_SEC", retry_backoff_base_sec)
        )

    def _pick_row(self, df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        for key in candidates:
            if key in df.index:
                return df.loc[key]
        return None

    def _try_prime_downloads(self, t) -> None:
        """Attempt to prime yfinance's lazy downloads to improve odds data is present."""
        for attr in ("financials", "balance_sheet", "cash_flow", "quarterly_financials", "quarterly_balance_sheet", "quarterly_cashflow"):
            try:
                _ = getattr(t, attr)
            except Exception:
                pass
        # Try newer API if present
        for mname in ("get_income_stmt", "get_balance_sheet", "get_cashflow"):
            m = getattr(t, mname, None)
            if callable(m):
                try:
                    # Some versions accept no args, others accept freq/quarterly flags
                    try:
                        m(freq="quarterly")
                    except Exception:
                        try:
                            m(quarterly=True)
                        except Exception:
                            m()
                except Exception:
                    pass

    def _load_quarterly_frames(self, t) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch quarterly income, balance sheet, and cash flow frames with fallbacks."""
        # Primary
        inc_raw = getattr(t, "quarterly_financials", None)
        bs_raw = getattr(t, "quarterly_balance_sheet", None)
        cf_raw = getattr(t, "quarterly_cashflow", None)

        # Fallbacks
        if (inc_raw is None or getattr(inc_raw, "empty", True)) and hasattr(t, "quarterly_income_stmt"):
            try:
                inc_raw = t.quarterly_income_stmt
            except Exception:
                pass
        if (inc_raw is None or getattr(inc_raw, "empty", True)) and hasattr(t, "income_stmt"):
            try:
                # Some versions expose quarterly via income_stmt with columns as periods
                inc = t.income_stmt
                # If annual, leave as is (we'll reject later if horizon not met)
                inc_raw = inc
            except Exception:
                pass
        return inc_raw, bs_raw, cf_raw

    def _build_per_ticker(self, ticker: str) -> Optional[Dict[str, object]]:
        t = yf.Ticker(ticker)
        # Optionally prime downloads to reduce rate-limit related empties
        self._try_prime_downloads(t)
        inc_raw, bs_raw, cf_raw = self._load_quarterly_frames(t)
        # Guard for missing data
        if inc_raw is None or bs_raw is None or cf_raw is None:
            return None
        if inc_raw.empty or bs_raw.empty or cf_raw.empty:
            return None

        df_inc_raw = inc_raw.transpose().sort_index()
        df_bs_raw = bs_raw.transpose().sort_index()
        df_cf_raw = cf_raw.transpose().sort_index()

        # Map to canonical columns
        def map_inc_row(r: pd.Series) -> Dict[str, float]:
            d = r.to_dict()
            # In yfinance, row labels typically:
            # 'Total Revenue', 'Cost Of Revenue', 'Selling General Administrative', 'Research Development',
            # 'Net Income', 'Interest Expense', 'Income Tax Expense', 'Interest Income'
            sales = d.get('Total Revenue')
            cogs = d.get('Cost Of Revenue')
            sgna = d.get('Selling General Administrative')
            rnd = d.get('Research Development')
            neti = d.get('Net Income')
            interest_exp = d.get('Interest Expense')
            income_tax_exp = d.get('Income Tax Expense')
            interest_inc = d.get('Interest Income')
            return {
                'sales': sales,
                'cogs': cogs,
                'opex': (sgna or 0.0) + (rnd or 0.0),
                'net_income': neti,
                'interest_expense': interest_exp,
                'income_tax_expense': income_tax_exp,
                'interest_income': interest_inc,
            }

        def map_bs_row(r: pd.Series) -> Dict[str, float]:
            d = r.to_dict()
            return {
                'cash': d.get('Cash And Cash Equivalents'),
                'st_investments': d.get('Short Term Investments'),
                'ar': d.get('Net Receivables') or d.get('Accounts Receivable'),
                'ap': d.get('Accounts Payable'),
                'inventory': d.get('Inventory') or d.get('Inventories'),
                'st_debt': d.get('Short Long Term Debt') or d.get('Short Term Debt'),
                'lt_debt': d.get('Long Term Debt'),
                'equity': d.get('Total Stockholder Equity') or d.get('Total Stockholders Equity'),
                'nfa': d.get('Property Plant Equipment') or d.get('Property Plant Equipment Net'),
            }

        def map_cf_row(r: pd.Series) -> Dict[str, float]:
            d = r.to_dict()
            capex = d.get('Capital Expenditures')
            divs = d.get('Cash Dividends Paid')
            return {
                'capex': float(capex) if capex is not None else 0.0,
                'dividends': float(divs) if divs is not None else 0.0,
            }

        df_inc = pd.DataFrame([map_inc_row(r) for _, r in df_inc_raw.iterrows()], index=df_inc_raw.index)
        df_bs = pd.DataFrame([map_bs_row(r) for _, r in df_bs_raw.iterrows()], index=df_bs_raw.index)
        df_cf = pd.DataFrame([map_cf_row(r) for _, r in df_cf_raw.iterrows()], index=df_cf_raw.index)

        # Align on common quarters
        idx = df_inc.index.intersection(df_bs.index).intersection(df_cf.index)
        df_inc = df_inc.loc[idx].sort_index()
        df_bs = df_bs.loc[idx].sort_index()
        df_cf = df_cf.loc[idx].sort_index()

        if len(df_inc) < self.horizon or len(df_bs) < self.horizon or len(df_cf) < self.horizon:
            return None

        # Take most recent horizon
        df_inc = df_inc.iloc[-self.horizon:]
        df_bs = df_bs.iloc[-self.horizon:]
        df_cf = df_cf.iloc[-self.horizon:]

        features_np = _build_features_panel(df_inc, df_bs, df_cf)
        targets_np = _df_to_targets(df_inc, df_bs)
        prev_dict = _df_to_prevstate(df_bs)
        return {
            'features': features_np,
            'targets': targets_np,
            'prev': prev_dict,
            'T': len(df_inc),
        }

    def all(self) -> Tuple[tf.Tensor, Policies, PrevState, Dict[str, tf.Tensor], List[str]]:
        per_ticker: Dict[str, Dict[str, object]] = {}
        for t in self.tickers:
            # polite delay before each request to reduce chance of rate-limit
            if self.per_request_delay_sec > 0:
                time.sleep(self.per_request_delay_sec)

            built = None
            last_err: Optional[Exception] = None
            for attempt in range(max(1, self.max_retries)):
                try:
                    built = self._build_per_ticker(t)
                except Exception as e:
                    last_err = e
                    built = None

                if built is not None and built.get('T', 0) >= self.horizon:
                    break  # success

                # Backoff before retry if not last attempt
                if attempt < max(1, self.max_retries) - 1:
                    sleep_s = self.retry_backoff_base_sec * (attempt + 1)
                    time.sleep(max(0.0, sleep_s))

            if built is None or built.get('T', 0) < self.horizon:
                # If we consistently failed for this ticker, just skip it.
                # Optionally log/print here if desired.
                continue
            per_ticker[t] = built
        tickers_kept = list(per_ticker.keys())
        if not tickers_kept:
            raise ValueError("No valid tickers with sufficient data found (yfinance)")

        # Stack batch (duplicate of YahooFinancialsLoader._stack_batch for independence)
        feats_list = [per_ticker[t]['features'] for t in tickers_kept]
        feats_np = np.stack(feats_list, axis=0)
        B, T, F = feats_np.shape
        feats_tf = tf.convert_to_tensor(feats_np, dtype=tf.float32)

        all_keys = set()
        for t in tickers_kept:
            all_keys |= set(per_ticker[t]['targets'].keys())
        targets_tf: Dict[str, tf.Tensor] = {}
        for key in all_keys:
            rows = []
            for t in tickers_kept:
                arr = per_ticker[t]['targets'].get(key)
                if arr is None:
                    arr = np.zeros((T, 1), dtype=np.float32)
                rows.append(arr)
            stacked = np.stack(rows, axis=0)
            targets_tf[key] = tf.convert_to_tensor(stacked, dtype=tf.float32)

        def stack_prev_field(field: str) -> tf.Tensor:
            vals = [[per_ticker[t]['prev'][field]] for t in tickers_kept]
            return tf.convert_to_tensor(np.array(vals, dtype=np.float32), dtype=tf.float32)
        prev_state = PrevState(
            cash=stack_prev_field('cash'),
            st_investments=stack_prev_field('st_investments'),
            st_debt=stack_prev_field('st_debt'),
            lt_debt=stack_prev_field('lt_debt'),
            ar=stack_prev_field('ar'),
            ap=stack_prev_field('ap'),
            inventory=stack_prev_field('inventory'),
            nfa=stack_prev_field('nfa'),
            equity=stack_prev_field('equity'),
        )

        def broadcast(val: float) -> tf.Tensor:
            return tf.ones((B, T, 1), dtype=tf.float32) * float(val)
        policies = Policies(
            min_cash_ratio=broadcast(self.defaults['min_cash_ratio']),
            payout_ratio=broadcast(self.defaults['payout_ratio']),
            tax_rate=broadcast(self.defaults['tax_rate']),
            inflation=broadcast(self.defaults['inflation']),
            real_rate=broadcast(self.defaults['real_rate']),
        )
        return feats_tf, policies, prev_state, targets_tf, tickers_kept


# ---------- CSV-based loader for manually downloaded statements ----------

class CSVLoader:
    """
    Loader for manually downloaded CSVs per ticker.

    Expected default filenames in `csv_dir`:
      - {TICKER}_financials.csv        (Income Statement)
      - {TICKER}_balance_sheet.csv
      - {TICKER}_cash_flow.csv

    Each CSV should have accounts as the first column (index) and period
    columns (e.g., '2024-06-30') across. This matches yfinance's typical
    DataFrame-to-CSV shape. We'll transpose to get [periods x accounts].

    Parameters allow custom filename patterns if needed.
    """

    def __init__(
        self,
        tickers: List[str],
        csv_dir: str,
        horizon_quarters: int = 8,
        financials_pattern: str = "{ticker}_financials.csv",
        balance_sheet_pattern: str = "{ticker}_balance_sheet.csv",
        cash_flow_pattern: str = "{ticker}_cash_flow.csv",
        min_cash_ratio_default: float = 0.05,
        payout_ratio_default: float = 0.20,
        tax_rate_default: float = 0.25,
        inflation_default: float = 0.02,
        real_rate_default: float = 0.01,
    ):
        self.tickers = tickers
        self.dir = csv_dir
        self.horizon = horizon_quarters
        self.financials_pattern = financials_pattern
        self.balance_sheet_pattern = balance_sheet_pattern
        self.cash_flow_pattern = cash_flow_pattern
        self.defaults = dict(
            min_cash_ratio=min_cash_ratio_default,
            payout_ratio=payout_ratio_default,
            tax_rate=tax_rate_default,
            inflation=inflation_default,
            real_rate=real_rate_default,
        )

    def _path(self, pattern: str, ticker: str) -> str:
        return os.path.join(self.dir, pattern.format(ticker=ticker))

    def _read_csv_matrix(self, path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        # Try to coerce column labels to datetimes if possible
        try:
            df.columns = pd.to_datetime(df.columns)
        except Exception:
            # If columns aren't dates, try rows as dates instead
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        return df

    def _build_per_ticker(self, ticker: str) -> Optional[Dict[str, object]]:
        fin_path = self._path(self.financials_pattern, ticker)
        bs_path = self._path(self.balance_sheet_pattern, ticker)
        cf_path = self._path(self.cash_flow_pattern, ticker)

        fin_raw = self._read_csv_matrix(fin_path)
        bs_raw = self._read_csv_matrix(bs_path)
        cf_raw = self._read_csv_matrix(cf_path)
        if fin_raw is None or bs_raw is None or cf_raw is None:
            return None

        # Transpose if accounts are rows and periods are columns
        df_inc_raw = fin_raw.transpose() if fin_raw.index.size > fin_raw.columns.size else fin_raw
        df_bs_raw = bs_raw.transpose() if bs_raw.index.size > bs_raw.columns.size else bs_raw
        df_cf_raw = cf_raw.transpose() if cf_raw.index.size > cf_raw.columns.size else cf_raw

        # Ensure period index is sorted ascending
        try:
            df_inc_raw = df_inc_raw.sort_index()
            df_bs_raw = df_bs_raw.sort_index()
            df_cf_raw = df_cf_raw.sort_index()
        except Exception:
            pass

        # Map to canonical columns (reuse yfinance-alias maps)
        def map_inc_row(r: pd.Series) -> Dict[str, float]:
            d = r.to_dict()
            sales = _pick_first_key(d, [
                'Total Revenue', 'totalRevenue', 'total_revenue', 'Revenue', 'revenue'
            ])
            cogs = _pick_first_key(d, [
                'Cost Of Revenue', 'costOfRevenue', 'cost_of_revenue', 'Cost of Goods Sold'
            ])
            sgna = _pick_first_key(d, [
                'Selling General Administrative', 'sellingGeneralAdministrative', 'SG&A'
            ])
            rnd = _pick_first_key(d, [
                'Research Development', 'researchDevelopment', 'R&D'
            ])
            neti = _pick_first_key(d, ['Net Income', 'netIncome', 'net_income'])
            interest_exp = _pick_first_key(d, ['Interest Expense', 'interestExpense'])
            income_tax_exp = _pick_first_key(d, ['Income Tax Expense', 'incomeTaxExpense'])
            interest_inc = _pick_first_key(d, ['Interest Income', 'interestIncome'])
            return {
                'sales': sales,
                'cogs': cogs,
                'opex': (sgna or 0.0) + (rnd or 0.0),
                'net_income': neti,
                'interest_expense': interest_exp,
                'income_tax_expense': income_tax_exp,
                'interest_income': interest_inc,
            }

        def map_bs_row(r: pd.Series) -> Dict[str, float]:
            d = r.to_dict()
            return {
                'cash': _pick_first_key(d, ['Cash And Cash Equivalents', 'cashAndCashEquivalents', 'cash']),
                'st_investments': _pick_first_key(d, ['Short Term Investments', 'shortTermInvestments']),
                'ar': _pick_first_key(d, ['Net Receivables', 'Accounts Receivable', 'accountsReceivable']),
                'ap': _pick_first_key(d, ['Accounts Payable', 'accountsPayable']),
                'inventory': _pick_first_key(d, ['Inventory', 'Inventories', 'inventory']),
                'st_debt': _pick_first_key(d, ['Short Long Term Debt', 'Short Term Debt', 'shortTermDebt']),
                'lt_debt': _pick_first_key(d, ['Long Term Debt', 'longTermDebt']),
                'equity': _pick_first_key(d, ['Total Stockholder Equity', 'Total Stockholders Equity', 'stockholdersEquity']),
                'nfa': _pick_first_key(d, ['Property Plant Equipment', 'Property Plant Equipment Net', 'propertyPlantEquipmentNet']),
            }

        def map_cf_row(r: pd.Series) -> Dict[str, float]:
            d = r.to_dict()
            capex = _pick_first_key(d, ['Capital Expenditures', 'capitalExpenditures'])
            divs = _pick_first_key(d, ['Cash Dividends Paid', 'dividendsPaid'])
            return {
                'capex': float(capex) if capex is not None else 0.0,
                'dividends': float(divs) if divs is not None else 0.0,
            }

        df_inc = pd.DataFrame([map_inc_row(r) for _, r in df_inc_raw.iterrows()], index=df_inc_raw.index)
        df_bs = pd.DataFrame([map_bs_row(r) for _, r in df_bs_raw.iterrows()], index=df_bs_raw.index)
        df_cf = pd.DataFrame([map_cf_row(r) for _, r in df_cf_raw.iterrows()], index=df_cf_raw.index)

        # Align and slice horizon
        idx = df_inc.index.intersection(df_bs.index).intersection(df_cf.index)
        if len(idx) == 0:
            return None
        df_inc = df_inc.loc[idx].sort_index()
        df_bs = df_bs.loc[idx].sort_index()
        df_cf = df_cf.loc[idx].sort_index()
        if len(df_inc) < self.horizon or len(df_bs) < self.horizon or len(df_cf) < self.horizon:
            return None
        df_inc = df_inc.iloc[-self.horizon:]
        df_bs = df_bs.iloc[-self.horizon:]
        df_cf = df_cf.iloc[-self.horizon:]

        features_np = _build_features_panel(df_inc, df_bs, df_cf)
        targets_np = _df_to_targets(df_inc, df_bs)
        prev_dict = _df_to_prevstate(df_bs)
        return {
            'features': features_np,
            'targets': targets_np,
            'prev': prev_dict,
            'T': len(df_inc),
        }

    def all(self) -> Tuple[tf.Tensor, Policies, PrevState, Dict[str, tf.Tensor], List[str]]:
        per_ticker: Dict[str, Dict[str, object]] = {}
        for t in self.tickers:
            built = self._build_per_ticker(t)
            if built is not None and built['T'] >= self.horizon:
                per_ticker[t] = built
        tickers_kept = list(per_ticker.keys())
        if not tickers_kept:
            raise ValueError("No valid tickers with sufficient data found (CSV loader)")

        feats_list = [per_ticker[t]['features'] for t in tickers_kept]
        feats_np = np.stack(feats_list, axis=0)
        B, T, F = feats_np.shape
        feats_tf = tf.convert_to_tensor(feats_np, dtype=tf.float32)

        all_keys = set()
        for t in tickers_kept:
            all_keys |= set(per_ticker[t]['targets'].keys())
        targets_tf: Dict[str, tf.Tensor] = {}
        for key in all_keys:
            rows = []
            for t in tickers_kept:
                arr = per_ticker[t]['targets'].get(key)
                if arr is None:
                    arr = np.zeros((T, 1), dtype=np.float32)
                rows.append(arr)
            stacked = np.stack(rows, axis=0)
            targets_tf[key] = tf.convert_to_tensor(stacked, dtype=tf.float32)

        def stack_prev_field(field: str) -> tf.Tensor:
            vals = [[per_ticker[t]['prev'][field]] for t in tickers_kept]
            return tf.convert_to_tensor(np.array(vals, dtype=np.float32), dtype=tf.float32)
        prev_state = PrevState(
            cash=stack_prev_field('cash'),
            st_investments=stack_prev_field('st_investments'),
            st_debt=stack_prev_field('st_debt'),
            lt_debt=stack_prev_field('lt_debt'),
            ar=stack_prev_field('ar'),
            ap=stack_prev_field('ap'),
            inventory=stack_prev_field('inventory'),
            nfa=stack_prev_field('nfa'),
            equity=stack_prev_field('equity'),
        )

        def broadcast(val: float) -> tf.Tensor:
            return tf.ones((B, T, 1), dtype=tf.float32) * float(val)
        policies = Policies(
            min_cash_ratio=broadcast(self.defaults['min_cash_ratio']),
            payout_ratio=broadcast(self.defaults['payout_ratio']),
            tax_rate=broadcast(self.defaults['tax_rate']),
            inflation=broadcast(self.defaults['inflation']),
            real_rate=broadcast(self.defaults['real_rate']),
        )

        return feats_tf, policies, prev_state, targets_tf, tickers_kept

# ---------- DummyData for tests ----------

class DummyData:

    """
    Synthetic generator for unit tests and offline bring-up.

    Shapes:
        B companies, T periods, F features each period.

    Keep this because:
    - CI shouldn't rely on external HTTP.
    - Unit tests should be deterministic.
    """

    def __init__(self, B=2, T=12, F=8):
        self.B, self.T, self.F = B, T, F

    def policies(self) -> Policies:
        B, T = self.B, self.T
        def ones(v):
            return tf.ones([B, T, 1], tf.float32) * v
        return Policies(
            inflation=ones(0.03),
            real_rate=ones(0.01),
            tax_rate=ones(0.25),
            min_cash_ratio=ones(0.05),
            payout_ratio=ones(0.20),
        )

    def prev(self) -> PrevState:
        B = self.B
        def v(x):
            return tf.ones([B, 1], tf.float32) * x
        return PrevState(
            cash=v(10.0),
            st_investments=v(0.0),
            st_debt=v(5.0),
            lt_debt=v(20.0),
            ar=v(8.0),
            ap=v(6.0),
            inventory=v(7.0),
            nfa=v(50.0),
            equity=v(44.0),
        )

    def features(self) -> tf.Tensor:
        return tf.random.uniform([self.B, self.T, self.F], 0.0, 1.0, tf.float32)

    def targets(self) -> Dict[str, tf.Tensor]:
        def r(lo, hi):
            return tf.random.uniform([self.B, self.T, 1], lo, hi, tf.float32)
        return {
            "sales":        r(50.0, 200.0),
            "cogs":         r(20.0, 150.0),
            "opex":         r(5.0, 50.0),
            "net_income":   r(-10.0, 30.0),
            "cash":         r(5.0, 40.0),
            "ar":           r(1.0, 20.0),
            "ap":           r(1.0, 20.0),
            "inventory":    r(1.0, 25.0),
            "st_debt":      r(0.0, 15.0),
            "lt_debt":      r(5.0, 50.0),
            "equity":       r(20.0, 80.0),
            "st_investments": r(0.0, 10.0),
            "nfa":            r(30.0, 90.0),
        }

