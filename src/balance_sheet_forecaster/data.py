from __future__ import annotations
from yahoofinancials import YahooFinancials
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import tensorflow as tf

from balance_sheet_forecaster.types import Policies, PrevState


# Safe extraction of data from Yahoo's nested dictionaries

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
    
    top_key = next(iter(yf_block.keys()), None)
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

def _safe_ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
        floor: float = 1e-6,
) -> np.ndarray:
    return np.divide(
        numerator, 
        np.maximum(np.abs(denominator), floor)
    )


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
    dividends_arr = df_cf["dividends_paid"].to_numpy(dtype=np.float32)

    # Derived features
    revenue_growth = np.zeros_like(sales_arr)
    revenue_growth[1:] = _safe_ratio(
        sales_arr[1:] - sales_arr[:-1],
        sales_arr[:-1]  # absolute value or not?
    )

    # Gross profit margin = 1 - COGS / Sales
    gross_margin = 1.0 - _safe_ratio(cogs_arr, sales_arr)

    # Opex ratio = Opex / Sales
    opex_ratio = _safe_ratio(opex_arr, sales_arr)

    # Leverage ratios
    debt_total = st_debt_arr + lt_debt_arr
    debt_to_equity = _safe_ratio(debt_total, equity_arr)

    # Working capital timing: realized DSO, DPO, DIO
    dso_days = _days_on_hand(ar_arr, sales_arr)
    dpo_days = _days_on_hand(ap_arr, cogs_arr)
    dio_days = _days_on_hand(inv_arr, cogs_arr)

    # Capex intensity = Capex / Sales (flip sign since capex is negative cash flow)
    capex_intensity = _safe_ratio(-capex_arr, sales_arr)

    # Payout ratio = Dividends / Net Income (flip sign since dividends are negative cash flow)
    payout_ratio = _safe_ratio(-dividends_arr, net_income_arr)

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
    )

    return feats.astype(np.float32)


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
        "sales": col(df_inc, "sales"),
        "cogs": col(df_inc, "cogs"),
        "opex": col(df_inc, "opex"),
        "net_income": col(df_inc, "net_income"),
        "cash": col(df_bs, "cash"),
        "st_investments": col(df_bs, "st_investments"),
        "ar": col(df_bs, "ar"),
        "ap": col(df_bs, "ap"),
        "inventory": col(df_bs, "inventory"),
        "st_debt": col(df_bs, "st_debt"),
        "lt_debt": col(df_bs, "lt_debt"),
        "nfa": col(df_bs, "nfa"),
        "equity": col(df_bs, "equity"),
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

        return {
            "sales": sales,
            "cogs": cogs,
            "opex": (sgna or 0.0) + (rnd or 0.0),
            "net_income": neti,
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
        }

    def map_cf_row(r: pd.Series) -> Dict[str, float]:
        d = r.to_dict()
        capex = _pick_first_key(d, ["capitalExpenditures", "capitalExpenditure"])
        divs  = _pick_first_key(d, ["dividendsPaid", "cashDividendsPaid"])
        return {
            "capex": float(capex),
            "dividends": float(divs),
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
    Loader pulls quarterly financial statements from Yahoo Finance
    for one or more tickers, engineers features, and packages everything
    into tensors that match the core model's expectations.

    Output dimensions:
        features: [B, T, F]
        policies: Policies with each field [B, T, 1]
        prev: PrevState with each field [B, 1]
        targets: Dict[str, tf.Tensor] with each field [B, T, 1]

    Notes:
        - We currently assume all tickers have the same number of historical quarters T.
          In practice, we may need to pad/truncate sequences to handle varying lengths.
        - We assume policies are broadcast constant over time for simplicity, 
          (can differ per ticket later).
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
        """
        Fetch quarterly income statement, balance sheet, and cash flow
        for all tickers using YahooFinancials.
        """

        yf = YahooFinancials(self.tickers)

        inc_q = yf.get_financial_stmts('quarterly', 'income')
        bs_q = yf.get_financial_stmts('quarterly', 'balance')
        cf_q = yf.get_financial_stmts('quarterly', 'cash')

        return inc_q, bs_q, cf_q
    
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

