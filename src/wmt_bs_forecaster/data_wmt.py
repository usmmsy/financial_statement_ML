from __future__ import annotations
"""Walmart CSV loader -> TF-native driver/policy tensors + previous state + targets.

Enhancements vs earlier scaffold:
1. Infers working capital days (DSO/DPO/DIO) from historical AR/AP/Inventory.
2. Constructs `PrevStateWMT` dataclass instead of raw dict.
3. Adds balance sheet targets (ar, ap, inventory, net_ppe, equity).
4. Aligns target naming with structural layer (`ebit` instead of operating_income).
5. Defensive NaN handling & clamping of inferred days.

Assumptions:
- CSV first column holds line item names; subsequent columns are chronological quarters.
- We take the next `horizon` columns as the forecast window and infer prev state from the column immediately before them (if available). If not, use first available.
- Equity decomposed as retained + paid-in (if paid-in absent, treat it as 0).

TODO (future): incorporate goodwill / other assets explicitly, multi-company batching (B>1).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import tensorflow as tf

from .types_wmt import DriversWMT, PoliciesWMT, PrevStateWMT


def _to_tf(x: np.ndarray, keep_time: bool = False) -> tf.Tensor:
    x = np.asarray(x, dtype=np.float32)
    if keep_time:
        return tf.convert_to_tensor(x.reshape(1, x.shape[0], 1))
    return tf.convert_to_tensor(x.reshape(1, 1))


def load_wmt_csvs(
    financials_csv: str,
    balance_csv: str,
    cashflow_csv: Optional[str] = None,
    horizon: int = 8,
    infer_wc_days: bool = True,
    clamp_days: tuple[float, float] = (0.0, 120.0),
) -> Tuple[DriversWMT, PoliciesWMT, PrevStateWMT, Dict[str, tf.Tensor]]:
    """Load Walmart quarterly CSVs.

    Returns:
        drivers: DriversWMT time-series tensors [B=1,T,1]
        policies: PoliciesWMT time-series tensors [1,T,1]
        prev_state: PrevStateWMT balances [1,1]
        targets: dict of target time-series we may fit (masked NaNs ignored by loss)

    Horizon:
        Select first `horizon` future/forecast columns after the first (name) column.
    Working Capital Days:
        If infer_wc_days: DSO = AR/Revenue*period_days etc. using same horizon slice.
    """
    fin = pd.read_csv(financials_csv)
    bs = pd.read_csv(balance_csv)
    cf = pd.read_csv(cashflow_csv) if cashflow_csv else None

    all_period_cols = list(fin.columns[1:])  # exclude name col
    horizon_cols = all_period_cols[:horizon]

    def series(df: pd.DataFrame, row_name: str) -> np.ndarray:
        s = df.loc[df.iloc[:, 0] == row_name, horizon_cols].values
        if s.size == 0:
            return np.full((len(horizon_cols),), np.nan, dtype=np.float32)
        return s.flatten().astype(np.float32)

    # Income statement lines
    revenue = np.nan_to_num(series(fin, "Total Revenue"), nan=0.0)
    cogs = np.nan_to_num(series(fin, "Cost Of Revenue"), nan=0.0)
    sga = np.nan_to_num(series(fin, "Selling General And Administration"), nan=0.0)
    operating_income = np.nan_to_num(series(fin, "Operating Income"), nan=0.0)  # used only as historical reference; target -> ebit

    # Balance sheet time-series for WC inference & targets
    ar_hist = series(bs, "Accounts Receivable")
    ap_hist = series(bs, "Accounts Payable")
    inv_hist = series(bs, "Inventory")
    net_ppe_hist = series(bs, "Net PPE")
    retained_hist = series(bs, "Retained Earnings")
    paid_in_hist = series(bs, "Additional Paid In Capital")
    equity_hist = retained_hist + np.nan_to_num(paid_in_hist, nan=0.0)

    # Phase 2 additional exogenous lines (paths) — treat missing as zeros
    other_current_assets_hist = np.nan_to_num(series(bs, "Other Current Assets"), nan=0.0)
    goodwill_hist = np.nan_to_num(series(bs, "Goodwill And Other Intangible Assets"), nan=0.0)
    other_non_current_assets_hist = np.nan_to_num(series(bs, "Other Non Current Assets"), nan=0.0)
    accrued_expenses_hist = np.nan_to_num(series(bs, "Current Accrued Expenses"), nan=0.0)
    tax_payable_hist = np.nan_to_num(series(bs, "Total Tax Payable"), nan=0.0)  # historical end balance (target only now)
    other_non_current_liab_hist = np.nan_to_num(series(bs, "Other Non Current Liabilities"), nan=0.0)
    minority_interest_hist = np.nan_to_num(series(bs, "Minority Interest"), nan=0.0)
    # treasury_stock dropped from schema; keep for target-only compatibility if needed
    # Additional commonly missing but impactful lines (try multiple aliases)
    def series_or_zero(names: list[str]) -> np.ndarray:
        vals = np.zeros((len(horizon_cols),), dtype=np.float32)
        hit = False
        for n in names:
            s = series(bs, n)
            if not np.all(np.isnan(s)):
                vals = np.nan_to_num(s, nan=0.0)
                hit = True
                break
        return np.nan_to_num(vals, nan=0.0)

    st_investments_hist = series_or_zero(["Short Term Investments", "Marketable Securities"])  # assets
    dividends_payable_hist = series_or_zero(["Dividends Payable"])  # historical end balance (target only now)
    # other_payables removed (redundant decomposition)
    # Removed explicit current deferred liabilities target; if present, fold into accrued later
    current_lease_hist = series_or_zero(["Current Capital Lease Obligation", "Current Operating Lease Liabilities"])  # current liability
    long_term_lease_hist = series_or_zero(["Long Term Capital Lease Obligation", "Non Current Operating Lease Liabilities"])  # non-current liability
    capital_stock_hist = series_or_zero(["Capital Stock", "Common Stock"])  # separate equity component
    fx_translation_hist = series(bs, "Foreign Currency Translation Adjustments")
    gains_losses_hist = series(bs, "Gains Losses Not Affecting Retained Earnings")
    other_equity_adj_hist = series(bs, "Other Equity Adjustments")
    # Accumulated OCI approximation: sum of available components (NaNs treated as 0)
    aoci_hist = np.nan_to_num(fx_translation_hist) + np.nan_to_num(gains_losses_hist) + np.nan_to_num(other_equity_adj_hist)

    # Opex ratio derived from SG&A / Revenue (NaNs -> 0)
    opex_ratio_arr = np.nan_to_num(np.divide(sga, revenue, out=np.zeros_like(sga), where=(revenue != 0)), nan=0.0)

    # Infer WC days if requested, else set priors.
    period_days = 365.0 / 4.0  # quarterly approximation
    def infer_days(balance: np.ndarray, flow: np.ndarray) -> np.ndarray:
        raw = np.divide(balance, flow, out=np.zeros_like(balance), where=(flow != 0)) * period_days
        lo, hi = clamp_days
        return np.clip(raw, lo, hi)

    if infer_wc_days:
        dso_days_arr = infer_days(ar_hist, revenue)
        dpo_days_arr = infer_days(ap_hist, cogs)
        dio_days_arr = infer_days(inv_hist, cogs)
    else:
        dso_days_arr = np.full_like(revenue, 30.0)
        dpo_days_arr = np.full_like(revenue, 45.0)
        dio_days_arr = np.full_like(revenue, 40.0)

    # Build DriversWMT
    # Cash flow derived drivers (optional) from cash flow statement if provided
    if cf is not None:
        # Normalize CF signs: provider may report outflows negative. We convert to positive magnitudes for outflows.
        def cf_series(name: str) -> np.ndarray:
            return np.nan_to_num(series(cf, name), nan=0.0)
        def cf_series_try(names: list[str]) -> np.ndarray:
            out = np.full((len(horizon_cols),), np.nan, dtype=np.float32)
            for n in names:
                s = series(cf, n)
                if s.size > 0 and not np.all(np.isnan(s)):
                    out = np.nan_to_num(s, nan=0.0)
                    break
            return np.nan_to_num(out, nan=0.0)
        change_in_tax_payable_hist = cf_series("Change In Tax Payable")
        change_in_accrued_expenses_hist = cf_series("Change In Accrued Expenses")
        cash_dividends_paid_raw = cf_series("Cash Dividends Paid")
        cash_dividends_paid_hist = np.abs(cash_dividends_paid_raw)  # outflow magnitude
        # Investing and FX/OCI related features (best-effort; may be zeros if rows absent)
        nbps_hist = cf_series("Net Business Purchase And Sale")
        net_invest_ps_hist = cf_series("Net Investment Purchase And Sale")
        net_other_investing_hist = cf_series("Net Other Investing Changes")
        fx_effect_hist = cf_series("Effect Of Exchange Rate Changes")
        gl_invest_hist = cf_series("Gain Loss On Investment Securities")
    # Equity financing: net common stock issuance
        # Prefer an explicit net line if available, else issuance minus repurchase magnitudes
        net_cs_hist_try = cf_series_try([
            "Issuance Of Common Stock, Net",
            "Common Stock Issued, Net",
            "Common Stock, Net"
        ])
        if np.allclose(net_cs_hist_try, 0.0):
            issuance = cf_series_try([
                "Issuance Of Common Stock",
                "Sale Of Common Stock",
                "Common Stock Issued"
            ])
            repurchase = cf_series_try([
                "Repurchase Of Common Stock",
                "Purchase Of Common Stock",
                "Common Stock Repurchased",
                "Payments For Repurchase Of Common Stock"
            ])
            net_common_stock_issuance_hist = np.nan_to_num(issuance, nan=0.0) - np.abs(np.nan_to_num(repurchase, nan=0.0))
        else:
            net_common_stock_issuance_hist = net_cs_hist_try
    else:
        change_in_tax_payable_hist = np.zeros_like(tax_payable_hist)
        change_in_accrued_expenses_hist = np.zeros_like(accrued_expenses_hist)
        cash_dividends_paid_hist = np.zeros_like(dividends_payable_hist)
        nbps_hist = np.zeros_like(dividends_payable_hist)
        net_invest_ps_hist = np.zeros_like(dividends_payable_hist)
        net_other_investing_hist = np.zeros_like(dividends_payable_hist)
        fx_effect_hist = np.zeros_like(dividends_payable_hist)
        gl_invest_hist = np.zeros_like(dividends_payable_hist)
        net_common_stock_issuance_hist = np.zeros_like(dividends_payable_hist)

    # Compute change in minority interest per period: first diff vs prev state, then in-sample diffs.
    if len(horizon_cols) > 0:
        # prev state's minority interest value (col before horizon) if available; else 0
        def latest_or_zero_all(row_name: str) -> float:
            arr = bs.loc[bs.iloc[:, 0] == row_name, all_period_cols].values
            return float(arr[0, 0]) if arr.size > 0 and not pd.isna(arr[0, 0]) else 0.0
        prev_minority_level = latest_or_zero_all("Minority Interest")
        mi_deltas = np.zeros_like(minority_interest_hist)
        if mi_deltas.size > 0:
            mi_deltas[0] = minority_interest_hist[0] - prev_minority_level
            if mi_deltas.size > 1:
                mi_deltas[1:] = minority_interest_hist[1:] - minority_interest_hist[:-1]
    else:
        mi_deltas = minority_interest_hist

    # Additional CF features used for deferred liability fallback
    if cf is not None:
        deferred_income_tax_hist = cf_series_try([
            "Deferred Income Tax",
            "Deferred Income Taxes",
            "Deferred Tax"
        ])
        other_non_cash_items_hist = cf_series_try([
            "Other Non Cash Items",
            "Other Non-Cash Items",
            "Other Noncash Items"
        ])
    else:
        deferred_income_tax_hist = np.zeros_like(dividends_payable_hist)
        other_non_cash_items_hist = np.zeros_like(dividends_payable_hist)

    # Preferred delta drivers for leases and deferred liabilities (zeros by default unless provided elsewhere)
    change_cur_lease_hist = np.zeros_like(dividends_payable_hist)
    change_lt_lease_hist = np.zeros_like(dividends_payable_hist)
    # removed change_in_current_deferred_liabilities driver

    drivers = DriversWMT(
        sales=_to_tf(revenue, keep_time=True),
        cogs=_to_tf(cogs, keep_time=True),
        capex=tf.zeros([1, len(horizon_cols), 1]),  # can override externally
        change_in_accrued_expenses=_to_tf(change_in_accrued_expenses_hist, keep_time=True),
        change_in_tax_payable=_to_tf(change_in_tax_payable_hist, keep_time=True),
        cash_dividends_paid=_to_tf(cash_dividends_paid_hist, keep_time=True),
        change_in_minority_interest=_to_tf(mi_deltas, keep_time=True),
        # Aggregate investing CF driver: sum of acquisitions and other investing flows
        aggregate_invest=_to_tf(nbps_hist + net_invest_ps_hist + net_other_investing_hist, keep_time=True),
        effect_of_exchange_rate_changes=_to_tf(fx_effect_hist, keep_time=True),
        gain_loss_on_investment_securities=_to_tf(gl_invest_hist, keep_time=True),
        deferred_income_tax=_to_tf(deferred_income_tax_hist, keep_time=True),
        other_non_cash_items=_to_tf(other_non_cash_items_hist, keep_time=True),
        change_in_current_capital_lease_obligation=_to_tf(change_cur_lease_hist, keep_time=True),
        change_in_long_term_capital_lease_obligation=_to_tf(change_lt_lease_hist, keep_time=True),
        net_common_stock_issuance=_to_tf(net_common_stock_issuance_hist, keep_time=True),
    )

    # Policies (basic priors; real rates zero => nominal ~= inflation)
    policies = PoliciesWMT(
        inflation=tf.zeros([1, len(horizon_cols), 1]),
        real_st_rate=tf.zeros([1, len(horizon_cols), 1]),
        real_lt_rate=tf.zeros([1, len(horizon_cols), 1]),
        tax_rate=tf.ones([1, len(horizon_cols), 1]) * 0.23,
        min_cash_ratio=tf.ones([1, len(horizon_cols), 1]) * 0.02,
        cash_coverage=None,
        lt_share_for_capex=tf.ones([1, len(horizon_cols), 1]) * 0.80,
        st_invest_spread=tf.ones([1, len(horizon_cols), 1]) * -0.02,
        debt_spread=tf.ones([1, len(horizon_cols), 1]) * 0.03,
        payout_ratio=tf.ones([1, len(horizon_cols), 1]) * 0.40,
        # Moved policy-like parameters
        dso_days=_to_tf(dso_days_arr, keep_time=True),
        dpo_days=_to_tf(dpo_days_arr, keep_time=True),
        dio_days=_to_tf(dio_days_arr, keep_time=True),
        opex_ratio=_to_tf(opex_ratio_arr, keep_time=True),
        depreciation_rate=tf.ones([1, len(horizon_cols), 1]) * 0.02,
        # Default coefficients (can be overridden by training/config)
        kappa_fx=tf.zeros([1, len(horizon_cols), 1]),
        kappa_unrealized=tf.zeros([1, len(horizon_cols), 1]),
        kappa_other=tf.zeros([1, len(horizon_cols), 1]),
        premium_ratio_goodwill=tf.zeros([1, len(horizon_cols), 1]),
        beta1_capex=tf.zeros([1, len(horizon_cols), 1]),
        # Single sensitivity of other NCA to aggregate investing CF
        beta2_net_invest=tf.zeros([1, len(horizon_cols), 1]),
        gamma_capital_stock=tf.zeros([1, len(horizon_cols), 1]),
        # Lease schedule coefficients (trainable/overridable)
        lease_addition_capex_coeff=tf.zeros([1, len(horizon_cols), 1]),
        lease_addition_sales_coeff=tf.zeros([1, len(horizon_cols), 1]),
        lease_addition_acq_coeff=tf.zeros([1, len(horizon_cols), 1]),
        lease_addition_other_invest_coeff=tf.zeros([1, len(horizon_cols), 1]),
        lease_addition_fx_coeff=tf.zeros([1, len(horizon_cols), 1]),
        lease_avg_remaining_term=tf.ones([1, len(horizon_cols), 1]) * 8.0,  # default ~2 years (8 quarters)
        lease_principal_payment_rate=tf.zeros([1, len(horizon_cols), 1]),
        lease_termination_sales_coeff=tf.zeros([1, len(horizon_cols), 1]),
        # Fallbacks for OCA and ONCL deltas
        omega_oca_sales=tf.zeros([1, len(horizon_cols), 1]),
        omega_oca_opex=tf.zeros([1, len(horizon_cols), 1]),
        psi_oncl_deferred_tax=tf.zeros([1, len(horizon_cols), 1]),
        psi_oncl_other_nc=tf.zeros([1, len(horizon_cols), 1]),
    )

    # Previous state selection: use earliest element of horizon slice as 'beg' approximation.
    # (If you want true previous quarter, supply a separate column slice.)
    def latest_or_zero(row_name: str) -> float:
        arr = bs.loc[bs.iloc[:, 0] == row_name, all_period_cols].values
        return float(arr[0, 0]) if arr.size > 0 and not pd.isna(arr[0, 0]) else 0.0

    # Build prev balances directly from reported line items (no residual equity plug)
    cash0 = latest_or_zero("Cash And Cash Equivalents")
    sti0 = latest_or_zero("Short Term Investments") or latest_or_zero("Marketable Securities")
    st0 = latest_or_zero("Current Debt")
    lt0 = latest_or_zero("Long Term Debt")
    ar0 = latest_or_zero("Accounts Receivable")
    ap0 = latest_or_zero("Accounts Payable")
    inv0 = latest_or_zero("Inventory")
    ppe0 = latest_or_zero("Net PPE")
    ret0 = latest_or_zero("Retained Earnings")
    paidin0 = latest_or_zero("Additional Paid In Capital")
    oca0 = latest_or_zero("Other Current Assets")
    gw0 = latest_or_zero("Goodwill And Other Intangible Assets")
    onca0 = latest_or_zero("Other Non Current Assets")
    accr0 = latest_or_zero("Current Accrued Expenses")
    tax0 = latest_or_zero("Total Tax Payable")
    oncl0 = latest_or_zero("Other Non Current Liabilities")
    aoci_fx0 = latest_or_zero("Foreign Currency Translation Adjustments")
    aoci_unr0 = latest_or_zero("Gains Losses Not Affecting Retained Earnings")
    aoci_oth0 = latest_or_zero("Other Equity Adjustments")
    aoci0 = aoci_fx0 + aoci_unr0 + aoci_oth0
    min0 = latest_or_zero("Minority Interest")
    # treasury stock dropped from prev state decomposition
    divpay0 = latest_or_zero("Dividends Payable")
    # other payables removed
    defrev0 = latest_or_zero("Current Deferred Liabilities") or latest_or_zero("Deferred Revenue") or latest_or_zero("Current Deferred Revenue")
    lease_cur0 = latest_or_zero("Current Capital Lease Obligation") or latest_or_zero("Current Operating Lease Liabilities")
    lease_lt0 = latest_or_zero("Long Term Capital Lease Obligation") or latest_or_zero("Non Current Operating Lease Liabilities")
    capstock0 = latest_or_zero("Capital Stock") or latest_or_zero("Common Stock")

    equity_reported0 = ret0 + paidin0  # equity decomposition (excluding AOCI, treasury, minority which are separate lines)

    # Fold any current deferred liabilities value into accrued expenses in prev state
    accr0 = accr0 + defrev0
    prev_state = PrevStateWMT(
        cash=_to_tf(np.array([cash0], dtype=np.float32)),
        st_investments=_to_tf(np.array([sti0], dtype=np.float32)),
        st_debt=_to_tf(np.array([st0], dtype=np.float32)),
        lt_debt=_to_tf(np.array([lt0], dtype=np.float32)),
        ar=_to_tf(np.array([ar0], dtype=np.float32)),
        ap=_to_tf(np.array([ap0], dtype=np.float32)),
        inventory=_to_tf(np.array([inv0], dtype=np.float32)),
        net_ppe=_to_tf(np.array([ppe0], dtype=np.float32)),
        equity=_to_tf(np.array([equity_reported0], dtype=np.float32)),
        retained_earnings=_to_tf(np.array([ret0], dtype=np.float32)),
        paid_in_capital=_to_tf(np.array([paidin0], dtype=np.float32)),
        other_current_assets=_to_tf(np.array([oca0], dtype=np.float32)),
        goodwill_intangibles=_to_tf(np.array([gw0], dtype=np.float32)),
        other_non_current_assets=_to_tf(np.array([onca0], dtype=np.float32)),
        accrued_expenses=_to_tf(np.array([accr0], dtype=np.float32)),
        tax_payable=_to_tf(np.array([tax0], dtype=np.float32)),
        other_non_current_liabilities=_to_tf(np.array([oncl0], dtype=np.float32)),
        aoci_fx=_to_tf(np.array([aoci_fx0], dtype=np.float32)),
        aoci_unrealized=_to_tf(np.array([aoci_unr0], dtype=np.float32)),
        aoci_other=_to_tf(np.array([aoci_oth0], dtype=np.float32)),
        aoci=_to_tf(np.array([aoci0], dtype=np.float32)),
        minority_interest=_to_tf(np.array([min0], dtype=np.float32)),
        dividends_payable=_to_tf(np.array([divpay0], dtype=np.float32)),
        current_capital_lease_obligation=_to_tf(np.array([lease_cur0], dtype=np.float32)),
        long_term_capital_lease_obligation=_to_tf(np.array([lease_lt0], dtype=np.float32)),
        capital_stock=_to_tf(np.array([capstock0], dtype=np.float32)),
    )

    # Targets: fit lines (NaNs are masked by loss)
    targets: Dict[str, tf.Tensor] = {
        "sales": _to_tf(revenue, keep_time=True),
        "cogs": _to_tf(cogs, keep_time=True),
        "opex": _to_tf(sga, keep_time=True),
        "ebit": _to_tf(operating_income, keep_time=True),  # align with structural layer attr name
        "ar": _to_tf(ar_hist, keep_time=True),
        "ap": _to_tf(ap_hist, keep_time=True),
        "inventory": _to_tf(inv_hist, keep_time=True),
        "net_ppe": _to_tf(net_ppe_hist, keep_time=True),
        "equity": _to_tf(equity_hist, keep_time=True),
        "other_current_assets": _to_tf(other_current_assets_hist, keep_time=True),
        "goodwill_intangibles": _to_tf(goodwill_hist, keep_time=True),
        "other_non_current_assets": _to_tf(other_non_current_assets_hist, keep_time=True),
        "accrued_expenses": _to_tf(accrued_expenses_hist, keep_time=True),
        "tax_payable": _to_tf(tax_payable_hist, keep_time=True),
        "other_non_current_liabilities": _to_tf(other_non_current_liab_hist, keep_time=True),
        "aoci": _to_tf(aoci_hist, keep_time=True),
        "minority_interest": _to_tf(minority_interest_hist, keep_time=True),
        "dividends_payable": _to_tf(dividends_payable_hist, keep_time=True),
    # other_payables removed
        "current_capital_lease_obligation": _to_tf(current_lease_hist, keep_time=True),
        "long_term_capital_lease_obligation": _to_tf(long_term_lease_hist, keep_time=True),
        "capital_stock": _to_tf(capital_stock_hist, keep_time=True),
    }

    return drivers, policies, prev_state, targets
