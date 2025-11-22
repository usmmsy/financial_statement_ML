import numpy as np
import pandas as pd
import tensorflow as tf

import pytest

from balance_sheet_forecaster.data import (
    _extract_quarterly_dict,
    _quarterly_dict_to_df,
    _safe_ratio,
    _days_on_hand,
    _build_features_panel,
    _df_to_targets,
    _df_to_prevstate,
    _normalize_ticker_frames,
    YahooFinancialsLoader,
)

# ---------- tiny helpers ----------
def _fake_quarter_block(key_name, ticker, rows):
    # rows: list of dicts like {"2023-03-31": {...}}
    return {key_name: {ticker: rows}}

def _mk_inc_df(dates):
    # minimal, but consistent with model’s expected column names
    return pd.DataFrame(
        {
            "sales": [100, 120, 110][: len(dates)],
            "cogs": [60, 70, 65][: len(dates)],
            "opex": [15, 18, 17][: len(dates)],
            "net_income": [20, 25, 22][: len(dates)],
        },
        index=pd.to_datetime(dates[:3]),
    )

def _mk_bs_df(dates):
    return pd.DataFrame(
        {
            "cash": [10, 12, 11][: len(dates)],
            "st_investments": [0, 1, 2][: len(dates)],
            "ar": [8, 9, 8.5][: len(dates)],
            "ap": [6, 6.5, 6.2][: len(dates)],
            "inventory": [7, 7.2, 7.1][: len(dates)],
            "st_debt": [5, 5.5, 5.2][: len(dates)],
            "lt_debt": [20, 19.5, 19.0][: len(dates)],
            "equity": [44, 45, 46][: len(dates)],
            "nfa": [50, 51, 52][: len(dates)],
        },
        index=pd.to_datetime(dates[:3]),
    )

def _mk_cf_df(dates):
    return pd.DataFrame(
        {
            "capex": [-6, -7, -5][: len(dates)],
            "dividends": [-3, -4, -2][: len(dates)],
        },
        index=pd.to_datetime(dates[:3]),
    )

# ---------- unit tests for helpers ----------

def test_extract_quarterly_dict_happy_and_empty():
    t = "MSFT"
    block = _fake_quarter_block(
        "balanceSheetHistoryQuarterly",
        t,
        [{"2023-03-31": {"k": 1}}, {"2022-12-31": {"k": 2}}],
    )
    out = _extract_quarterly_dict(block, t)
    assert set(out.keys()) == {"2023-03-31", "2022-12-31"}
    assert out["2023-03-31"]["k"] == 1

    # unknown ticker -> empty
    assert _extract_quarterly_dict(block, "AAPL") == {}

def test_quarterly_dict_to_df_sorts_and_indexes():
    d = {
        "2023-06-30": {"a": 1, "b": 2},
        "2023-03-31": {"a": 3, "b": 4},
    }
    df = _quarterly_dict_to_df(d)
    assert list(df.index) == sorted(df.index)  # ascending
    assert set(df.columns) == {"a", "b"}

def test_safe_ratio_and_days_on_hand_numerics():
    num = np.array([10.0, 0.0, -5.0], np.float32)
    den = np.array([2.0, 0.0, 0.0], np.float32)
    r = _safe_ratio(num, den, floor=1e-6)
    # division by zero is floored, finite results
    assert np.all(np.isfinite(r))

    bal = np.array([100.0, 50.0], np.float32)
    flow = np.array([20.0, 10.0], np.float32)
    days = _days_on_hand(bal, flow)
    # ~ 100/20*365 = 1825, 50/10*365 = 1825
    assert np.allclose(days, np.array([1825.0, 1825.0], np.float32), rtol=1e-6, atol=1e-6)

def test_build_features_panel_shapes_and_types():
    dates = ["2022-12-31", "2023-03-31", "2023-06-30"]
    df_inc = _mk_inc_df(dates)
    df_bs = _mk_bs_df(dates)
    df_cf = _mk_cf_df(dates)

    feats = _build_features_panel(df_inc, df_bs, df_cf)
    # we defined 10 features in _build_features_panel
    assert feats.shape == (3, 10)
    assert feats.dtype == np.float32

def test_df_to_targets_and_prevstate_shapes():
    dates = ["2022-12-31", "2023-03-31", "2023-06-30"]
    df_inc = _mk_inc_df(dates)
    df_bs = _mk_bs_df(dates)

    targets = _df_to_targets(df_inc, df_bs)
    # each is [T,1]
    for k, v in targets.items():
        assert isinstance(v, np.ndarray)
        assert v.shape == (3, 1)
        assert v.dtype == np.float32

    prev = _df_to_prevstate(df_bs)
    # floats extracted from last row
    for key in ["cash", "st_investments", "st_debt", "lt_debt", "ar", "ap", "inventory", "nfa", "equity"]:
        assert isinstance(prev[key], float)

def test_normalize_ticker_frames_aligns_indices_and_columns():
    t = "AAPL"
    # Build yahoo-like nested blocks
    inc_block = _fake_quarter_block(
        "incomeStatementHistoryQuarterly",
        t,
        [
            {"2023-03-31": {"totalRevenue": 100, "costOfRevenue": 60, "researchDevelopment": 5, "sellingGeneralAdministrative": 10, "netIncome": 20}},
            {"2022-12-31": {"totalRevenue": 120, "costOfRevenue": 70, "researchDevelopment": 6, "sellingGeneralAdministrative": 12, "netIncome": 25}},
        ],
    )
    bs_block = _fake_quarter_block(
        "balanceSheetHistoryQuarterly",
        t,
        [
            {"2023-03-31": {"cashAndCashEquivalents": 10, "shortTermInvestments": 0, "netReceivables": 8, "accountsPayable": 6, "inventory": 7, "shortTermDebt": 5, "longTermDebt": 20, "totalStockholderEquity": 44, "propertyPlantEquipmentNet": 50}},
            {"2022-12-31": {"cashAndCashEquivalents": 12, "shortTermInvestments": 1, "netReceivables": 9, "accountsPayable": 6.5, "inventory": 7.2, "shortTermDebt": 5.5, "longTermDebt": 19.5, "totalStockholderEquity": 45, "propertyPlantEquipmentNet": 51}},
        ],
    )
    cf_block = _fake_quarter_block(
        "cashflowStatementHistoryQuarterly",
        t,
        [
            {"2023-03-31": {"capitalExpenditures": -6, "dividendsPaid": -3}},
            {"2022-12-31": {"capitalExpenditures": -7, "dividendsPaid": -4}},
        ],
    )

    df_inc, df_bs, df_cf = _normalize_ticker_frames(t, inc_block, bs_block, cf_block)
    # common dates intersection should be the two given quarters
    assert len(df_inc) == len(df_bs) == len(df_cf) == 2
    for cols in [("sales", "cogs", "opex", "net_income"), ("cash", "st_investments", "ar", "ap", "inventory", "st_debt", "lt_debt", "equity", "nfa"), ("capex", "dividends")]:
        for c in cols:
            assert c in (df_inc.columns if "sales" in cols else df_bs.columns if "cash" in cols else df_cf.columns)

def test_stack_batch_shapes_without_network(monkeypatch):
    # Synthesize per-ticker artifacts as _stack_batch expects.
    tickers = ["A", "B"]
    T, F = 3, 10

    per_ticker = {}
    for t in tickers:
        # features [T,F]
        feats = np.random.RandomState(0).randn(T, F).astype(np.float32)
        # targets dict of [T,1]
        targets = {
            "sales": np.ones((T, 1), np.float32),
            "cogs": np.full((T, 1), 0.5, np.float32),
            "cash": np.linspace(1, 3, T, dtype=np.float32).reshape(-1, 1),
        }
        # prev dict of floats
        prev = {
            "cash": 1.0, "st_investments": 0.0, "st_debt": 2.0, "lt_debt": 5.0,
            "ar": 0.5, "ap": 0.4, "inventory": 0.3, "nfa": 10.0, "equity": 6.0,
        }
        per_ticker[t] = {"features": feats, "targets": targets, "prev": prev, "T": T}

    loader = YahooFinancialsLoader(tickers=tickers, horizon_quarters=T)

    feats_tf, policies, prev_state, targets_tf, tickers_kept = loader._stack_batch(per_ticker)

    # shapes
    assert feats_tf.shape == (len(tickers), T, F)
    for field in ["inflation", "real_rate", "tax_rate", "min_cash_ratio", "payout_ratio"]:
        v = getattr(policies, field)
        assert v.shape == (len(tickers), T, 1)

    for field in ["cash", "st_investments", "st_debt", "lt_debt", "ar", "ap", "inventory", "nfa", "equity"]:
        v = getattr(prev_state, field)
        assert v.shape == (len(tickers), 1)

    # targets present and properly stacked
    assert set(targets_tf.keys()) == {"sales", "cogs", "cash"}
    for v in targets_tf.values():
        assert v.shape == (len(tickers), T, 1)

    assert tickers_kept == tickers
