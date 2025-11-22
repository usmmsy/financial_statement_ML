"""Calibrate WMT policies against historical quarters using the structural model.

This script assumes that scripts/estimate_wmt.py has already been run and that
wmt_policies_forward.npz and wmt_driver_params.npz exist under data/models.

We:
- Load initial PoliciesWMT (treated as constants) and then promote a subset of
  scalar schedules (gross_margin, opex_ratio, depreciation_rate, dso/dpo/dio,
  tax_rate, payout_ratio) to trainable TensorFlow variables.
- Build historical DriversWMT and PrevStateWMT from the WMT quarterly CSVs for
  the four training quarters (2024-07-31 .. 2025-04-30).
- Run the WMT StructuralLayer over those quarters, reconstructing income
  statement and balance sheet.
- Minimize a loss that combines:
    * accounting identity gap (Assets - LiabEq)
    * fit to selected BS lines (cash, AR, AP, inventory, net PPE, equity)
    * fit to selected IS lines (sales, gross profit, opex, net income)
- Save the calibrated policy schedules back to data/models.

This is intentionally lightweight and uses a small number of steps; refine
weights, learning rate, and loss composition as needed.
"""
from __future__ import annotations
import pathlib
from typing import Dict, List

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

from wmt_bs_forecaster.types_wmt import (
    PoliciesWMT,
    DriversWMT,
    PrevStateWMT,
    StatementsWMT,
)
from wmt_bs_forecaster.accounting_wmt import StructuralLayer

try:
    import scripts.estimate_wmt as _est
except ModuleNotFoundError:
    import estimate_wmt as _est

learn_gross_margin = _est.learn_gross_margin
learn_opex_ratio = _est.learn_opex_ratio
learn_depreciation_rate = _est.learn_depreciation_rate
learn_working_capital_days = _est.learn_working_capital_days
learn_tax_rate = _est.learn_tax_rate
learn_payout_ratio = _est.learn_payout_ratio
load_wmt_quarterlies = _est.load_wmt_quarterlies
extract_ppe_prev = _est.extract_ppe_prev
learn_basic_policies = _est.learn_basic_policies

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "retail_csv" / "WMT_quarterly"
MODEL_DIR = ROOT / "data" / "models"

BAL_PATH = DATA_DIR / "WMT_quarterly_balance_sheet.csv"
FIN_PATH = DATA_DIR / "WMT_quarterly_financials.csv"
CF_PATH = DATA_DIR / "WMT_quarterly_cash_flow.csv"

TRAIN_COLS = [
    "2024-07-31",
    "2024-10-31",
    "2025-01-31",
    "2025-04-30",
]

EPS = 1e-8


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


def load_quarterlies_train() -> Dict[str, pd.DataFrame]:
    bal = pd.read_csv(BAL_PATH, index_col=0)
    fin = pd.read_csv(FIN_PATH, index_col=0)
    cf = pd.read_csv(CF_PATH, index_col=0)
    common_cols = sorted(set(bal.columns) & set(fin.columns) & set(cf.columns))
    bal = bal[common_cols]
    fin = fin[common_cols]
    cf = cf[common_cols]
    cols = [c for c in TRAIN_COLS if c in bal.columns]
    if len(cols) < 2:
        raise ValueError(
            f"Not enough training quarters with full data. Found {cols}, available {list(bal.columns)}"
        )
    return {"bal": bal[cols], "fin": fin[cols], "cf": cf[cols], "cols": cols}


def load_initial_policies() -> PoliciesWMT:
    npz = np.load(MODEL_DIR / "wmt_policies_forward.npz")

    def t(name: str) -> tf.Tensor:
        return tf.convert_to_tensor(npz[name], dtype=tf.float32)

    kw = dict(
        inflation=t("inflation"),
        real_st_rate=t("real_st_rate"),
        real_lt_rate=t("real_lt_rate"),
        tax_rate=t("tax_rate"),
        payout_ratio=t("payout_ratio"),
        min_cash_ratio=t("min_cash_ratio"),
        lt_share_for_capex=t("lt_share_for_capex"),
        st_invest_spread=t("st_invest_spread"),
        debt_spread=t("debt_spread"),
        dso_days=t("dso_days"),
        dpo_days=t("dpo_days"),
        dio_days=t("dio_days"),
        opex_ratio=t("opex_ratio"),
        depreciation_rate=t("depreciation_rate"),
        gross_margin=t("gross_margin"),
        premium_ratio_goodwill=t("premium_ratio_goodwill"),
        beta1_capex=t("beta1_capex"),
        beta2_net_invest=t("beta2_net_invest"),
    )

    # Optional OCA/ONCL fallback coefficients and lease parameters may or may not be
    # present depending on when policies were estimated. Load them defensively.
    for name in [
        "omega_oca_sales",
        "omega_oca_opex",
        "psi_oncl_deferred_tax",
        "psi_oncl_other_nc",
        "lease_addition_capex_coeff",
        "lease_addition_sales_coeff",
        "lease_avg_remaining_term",
        "lease_principal_payment_rate",
        "gamma_capital_stock",
    ]:
        if name in npz.files:
            kw[name] = t(name)

    return PoliciesWMT(**kw)


def build_prev_state_train(bal: pd.DataFrame, cols: List[str]) -> PrevStateWMT:
    # Use the period immediately before the first training quarter as prev.
    # If missing, approximate using the first training column.
    all_cols = list(bal.columns)
    first_col = cols[0]
    if first_col not in all_cols:
        raise KeyError(f"First training column {first_col!r} not in balance sheet columns {all_cols}")
    first_idx = all_cols.index(first_col)
    prev_col = all_cols[first_idx - 1] if first_idx > 0 else first_col
    b_prev = bal[prev_col]

    def row_like(token: str, default: float = 0.0) -> float:
        for idx, val in b_prev.items():
            if token in idx.lower():
                return float(val)
        return default

    cash = row_like("cash and cash equivalents")
    st_inv = 0.0
    ar = row_like("accounts receivable")
    inventory = row_like("inventory")
    net_ppe = row_like("net ppe")
    st_debt = row_like("current debt")
    lt_debt = row_like("long term debt")
    ap = row_like("accounts payable")
    equity = row_like("stockholders equity")

    other_current_assets = row_like("other current assets")
    goodwill = row_like("goodwill")
    other_nca = row_like("other non current assets")
    accrued_expenses = row_like("current accrued expenses")
    tax_payable = row_like("total tax payable")
    other_ncl = row_like("other non current liabilities")
    cur_lease = row_like("current capital lease obligation")
    lt_lease = row_like("long term capital lease obligation")
    div_payable = row_like("dividends payable")
    aoci_total = row_like("gains losses not affecting retained earnings")
    minority = row_like("minority interest")
    cap_stock = row_like("capital stock")

    def scalar(x: float) -> tf.Tensor:
        return tf.convert_to_tensor([[x]], dtype=tf.float32)

    return PrevStateWMT(
        cash=scalar(cash),
        st_investments=scalar(st_inv),
        ar=scalar(ar),
        inventory=scalar(inventory),
        net_ppe=scalar(net_ppe),
        st_debt=scalar(st_debt),
        lt_debt=scalar(lt_debt),
        ap=scalar(ap),
        equity=scalar(equity),
        other_current_assets=scalar(other_current_assets),
        goodwill_intangibles=scalar(goodwill),
        other_non_current_assets=scalar(other_nca),
        accrued_expenses=scalar(accrued_expenses),
        tax_payable=scalar(tax_payable),
        other_non_current_liabilities=scalar(other_ncl),
        current_capital_lease_obligation=scalar(cur_lease),
        long_term_capital_lease_obligation=scalar(lt_lease),
        dividends_payable=scalar(div_payable),
        aoci=scalar(aoci_total),
        minority_interest=scalar(minority),
        capital_stock=scalar(cap_stock),
    )


def build_historical_drivers(bal: pd.DataFrame, fin: pd.DataFrame, cf: pd.DataFrame, cols: List[str]) -> DriversWMT:
    """Construct historical DriversWMT from financials and cash flow statements.

    For calibration we:
    - Use realized Total Revenue as DriversWMT.sales.
    - Let StructuralLayer derive COGS via gross_margin (so cogs path is zeros).
    - Populate key cash-flow-based drivers for WC, investing, AOCI, and equity
      dynamics directly from the vendor cash flow statement.
    """

    # Sales
    rev_row = _find_row(fin, "Total Revenue")
    if rev_row is None:
        raise ValueError("Total Revenue row not found for historical drivers.")
    sales_hist = fin.loc[rev_row, cols].to_numpy(dtype=np.float32)
    sales_tf = tf.convert_to_tensor(sales_hist.reshape(1, len(cols), 1), dtype=tf.float32)

    cogs_tf = tf.zeros_like(sales_tf)

    # Placeholder capex (kept zero; StructuralLayer uses depreciation_rate and
    # its own capex logic for calibration).
    capex_tf = tf.zeros_like(sales_tf)

    # Helper to extract a CF row as raw numpy array, cleaning NaNs to zero
    def cf_series_raw(phrase: str) -> np.ndarray:
        r = _find_row(cf, phrase)
        if r is None:
            return np.zeros(len(cols), dtype=np.float32)
        vals = cf.loc[r, cols].to_numpy(dtype=np.float32)
        return np.nan_to_num(vals, nan=0.0)

    # Helper to extract a CF row as [1, T, 1] tensor or zeros if missing
    def cf_series(phrase: str) -> tf.Tensor:
        vals = cf_series_raw(phrase)
        return tf.convert_to_tensor(vals.reshape(1, len(cols), 1), dtype=tf.float32)

    # Operating / WC-related deltas
    change_in_accrued_expenses = cf_series("Change In Accrued Expense")
    change_in_tax_payable = cf_series("Change In Income Tax Payable")

    # Deferred tax and other non-cash items (used for other non-current liabilities)
    deferred_income_tax = cf_series("Deferred Income Tax")
    other_non_cash_items = cf_series("Other Non Cash Items")
    # Investing drivers: aggregate sparse lines into a single net acquisitions & investments series.
    nbps_raw = cf_series_raw("Net Business Purchase And Sale")
    ninv_raw = cf_series_raw("Net Investment Purchase And Sale")
    noth_raw = cf_series_raw("Net Other Investing Changes")
    aggregate_invest_vals = nbps_raw + ninv_raw + noth_raw
    aggregate_invest = tf.convert_to_tensor(
        aggregate_invest_vals.reshape(1, len(cols), 1), dtype=tf.float32
    )
    effect_of_exchange_rate_changes = cf_series("Effect Of Exchange Rate Changes")
    gain_loss_on_investment_securities = cf_series("Gain Loss On Investment Securities")

    # Financing / equity-related drivers
    cash_dividends_paid = cf_series("Cash Dividends Paid")
    net_common_stock_issuance = cf_series("Net Common Stock Issuance")

    # AOCI total delta driver: derive from balance sheet line
    # "Gains Losses Not Affecting Retained Earnings" as ΔAOCI_t = A_t - A_{t-1}.
    # Use the same columns as training quarters.
    aoci_row = _find_row(bal, "Gains Losses Not Affecting Retained Earnings")
    if aoci_row is not None:
        aoci_series = bal.loc[aoci_row, cols].to_numpy(dtype=np.float32)
        # prev_aoci corresponds to the period immediately before the first
        # training quarter; when missing we approximate by reusing the first
        # training value.
        all_cols = list(bal.columns)
        first_col = cols[0]
        first_idx = all_cols.index(first_col)
        if first_idx > 0:
            prev_col = all_cols[first_idx - 1]
            prev_aoci_val = float(bal.loc[aoci_row, prev_col])
        else:
            prev_aoci_val = float(aoci_series[0])
        delta_vals = np.empty_like(aoci_series)
        # First training delta uses prev_aoci
        delta_vals[0] = aoci_series[0] - prev_aoci_val
        # Subsequent deltas are simple differences
        if len(aoci_series) > 1:
            delta_vals[1:] = aoci_series[1:] - aoci_series[:-1]
        change_in_aoci = tf.convert_to_tensor(
            delta_vals.reshape(1, len(cols), 1), dtype=tf.float32
        )
    else:
        change_in_aoci = None

    return DriversWMT(
        sales=sales_tf,
        cogs=cogs_tf,
        capex=capex_tf,
        change_in_accrued_expenses=change_in_accrued_expenses,
        change_in_tax_payable=change_in_tax_payable,
        deferred_income_tax=deferred_income_tax,
        other_non_cash_items=other_non_cash_items,
        aggregate_invest=aggregate_invest,
        effect_of_exchange_rate_changes=effect_of_exchange_rate_changes,
        gain_loss_on_investment_securities=gain_loss_on_investment_securities,
        cash_dividends_paid=cash_dividends_paid,
        net_common_stock_issuance=net_common_stock_issuance,
        change_in_aoci=change_in_aoci,
    )


def make_trainable_policies(base: PoliciesWMT, T_hist: int) -> Dict[str, tf.Variable]:
    """Create scalar trainable variables for selected policy paths.

    We assume the base PoliciesWMT has shape [1, T_hist+H, 1]. We only calibrate
    the first T_hist steps and keep forward extension constant.
    """

    def avg_scalar(x: tf.Tensor) -> float:
        x_hist = x[:, :T_hist, :]
        return float(tf.reduce_mean(x_hist).numpy())

    vars_dict: Dict[str, tf.Variable] = {}
    # Core schedules that shape the income statement and working capital.
    core_policy_names = [
        "gross_margin",
        "opex_ratio",
        "depreciation_rate",
        "dso_days",
        "dpo_days",
        "dio_days",
        "tax_rate",
        "payout_ratio",
        # Cash and funding policies that strongly influence debt levels.
        "min_cash_ratio",
        "lt_share_for_capex",
        # "st_invest_spread",
    ]
    # Investing-related coefficients controlling goodwill and other NCA.
    investing_names = [
        "premium_ratio_goodwill",
        "beta1_capex",
        "beta2_net_invest",
    ]
    # Capital stock sensitivity to net common stock issuance.
    gamma_names = [
        "gamma_capital_stock",
    ]
    # Lease-related schedule parameters.
    lease_names = [
        "lease_addition_capex_coeff",
        "lease_addition_sales_coeff",
        "lease_avg_remaining_term",
    ]
    # Other current assets fallback coefficients.
    oca_names = [
        "omega_oca_sales",
        "omega_oca_opex",
    ]
    # Other non-current liabilities fallback coefficients.
    oncl_names = [
        "psi_oncl_deferred_tax",
        "psi_oncl_other_nc",
    ]

    for name in core_policy_names + investing_names + gamma_names + lease_names + oca_names + oncl_names:
        base_tensor = getattr(base, name)
        if base_tensor is None:
            continue
        init_val = avg_scalar(base_tensor)
        vars_dict[name] = tf.Variable(init_val, dtype=tf.float32, name=f"theta_{name}")

    return vars_dict


def apply_trainable_policies(base: PoliciesWMT, theta: Dict[str, tf.Variable], T_hist: int) -> PoliciesWMT:
    """Return a new PoliciesWMT with first T_hist periods replaced by theta.

    This keeps other levers (macro, liquidity, spreads, etc.) fixed.

    We also clamp effective values into economically sensible ranges to avoid
    NaNs/Inf during calibration.
    """

    def extend_scalar(var_name: str, orig: tf.Tensor, lo: float | None = None, hi: float | None = None) -> tf.Tensor:
        if var_name not in theta:
            return orig[:, :T_hist, :]
        v = theta[var_name]
        v_eff = v
        if lo is not None or hi is not None:
            v_eff = tf.clip_by_value(v_eff,
                                     lo if lo is not None else v_eff,
                                     hi if hi is not None else v_eff)
        return tf.ones_like(orig[:, :T_hist, :]) * v_eff

    # Slice to historical window and overwrite the selected ones
    return PoliciesWMT(
        inflation=base.inflation[:, :T_hist, :],
        real_st_rate=base.real_st_rate[:, :T_hist, :],
        real_lt_rate=base.real_lt_rate[:, :T_hist, :],
        tax_rate=extend_scalar("tax_rate", base.tax_rate, 0.0, 0.5),
        payout_ratio=extend_scalar("payout_ratio", base.payout_ratio, 0.0, 1.0),
        min_cash_ratio=extend_scalar("min_cash_ratio", base.min_cash_ratio, 0.0, 0.10),
        lt_share_for_capex=extend_scalar("lt_share_for_capex", base.lt_share_for_capex, 0.0, 1.0),
        st_invest_spread=base.st_invest_spread[:, :T_hist, :],
        debt_spread=base.debt_spread[:, :T_hist, :],
        dso_days=extend_scalar("dso_days", base.dso_days, 0.0, 120.0),
        dpo_days=extend_scalar("dpo_days", base.dpo_days, 0.0, 180.0),
        dio_days=extend_scalar("dio_days", base.dio_days, 0.0, 200.0),
        opex_ratio=extend_scalar("opex_ratio", base.opex_ratio, -0.8, 0.8),
        depreciation_rate=extend_scalar("depreciation_rate", base.depreciation_rate, 0.0, 0.5),
        gross_margin=extend_scalar("gross_margin", base.gross_margin, 0.0, 0.9),
        # Investing-related coefficients: keep unclipped to allow the model to
        # discover appropriate signs/magnitudes, but still treat them as
        # time-constant scalars over the historical window.
        premium_ratio_goodwill=extend_scalar("premium_ratio_goodwill", base.premium_ratio_goodwill),
        beta1_capex=extend_scalar("beta1_capex", base.beta1_capex),
        beta2_net_invest=extend_scalar("beta2_net_invest", base.beta2_net_invest),
        gamma_capital_stock=extend_scalar("gamma_capital_stock", base.gamma_capital_stock),
        # Lease-related parameters: lightly constrained for numerical stability.
        lease_addition_capex_coeff=extend_scalar(
            "lease_addition_capex_coeff", base.lease_addition_capex_coeff, 0.0, 0.5
        ),
        lease_avg_remaining_term=extend_scalar(
            "lease_avg_remaining_term", base.lease_avg_remaining_term, 4.0, 40.0
        ),
        lease_addition_sales_coeff=extend_scalar(
            "lease_addition_sales_coeff", base.lease_addition_sales_coeff, 0.0, 0.5
        ),
        # OCA coefficients: keep unclipped for now (magnitudes are typically
        # small relative to flows), but still treat them as time-constant
        # scalars over the historical window.
        omega_oca_sales=extend_scalar("omega_oca_sales", base.omega_oca_sales),
        omega_oca_opex=extend_scalar("omega_oca_opex", base.omega_oca_opex),
        # ONCL coefficients: likewise left unclipped; their scale is moderated
        # by the magnitude of deferred tax and other non-cash drivers.
        psi_oncl_deferred_tax=extend_scalar("psi_oncl_deferred_tax", base.psi_oncl_deferred_tax),
        psi_oncl_other_nc=extend_scalar("psi_oncl_other_nc", base.psi_oncl_other_nc),
    )


def calibration_loss(stmts: StatementsWMT, bal: pd.DataFrame, fin: pd.DataFrame, cols: List[str]) -> tf.Tensor:
    """Combined identity + data-fit loss over training quarters.

    All components use *relative* soft penalties, reported as RMSE of relative
    errors for more direct interpretability:
    - Identity: relative RMSE between Assets and Liab+Equity.
    - Balance sheet fit: relative RMSE on selected lines.
    - Income statement fit: relative RMSE on selected lines.
    """

    def rel_rmse(model: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """Root-mean-squared relative error between model and target.

        Rough interpretation: if rel_rmse ~= 0.1, typical error is about 10%.
        """
        denom = tf.maximum(tf.maximum(tf.abs(target), tf.abs(model)), 1e-6)
        gap_rel = (model - target) / denom
        return tf.sqrt(tf.reduce_mean(tf.square(gap_rel)))

    # Identity gap (relative)
    assets = stmts.assets
    liab_eq = stmts.liab_plus_equity
    loss_identity = rel_rmse(assets, liab_eq)

    # Map pandas history into tensors of shape [1, T, 1]
    def hist_series(df: pd.DataFrame, row_tokens: str) -> tf.Tensor | None:
        r = _find_row(df, row_tokens)
        if r is None:
            return None
        vals = df.loc[r, cols].to_numpy(dtype=np.float32)
        return tf.convert_to_tensor(vals.reshape(1, len(cols), 1), dtype=tf.float32)

    # BS targets
    cash_t = hist_series(bal, "Cash And Cash Equivalents")  # cash and cash equivalents
    ar_t = hist_series(bal, "Accounts Receivable")
    ap_t = hist_series(bal, "Accounts Payable")
    inv_t = hist_series(bal, "Inventory")
    oca_t = hist_series(bal, "Other Current Assets")
    ppe_t = hist_series(bal, "Net PPE")
    eq_t = hist_series(bal, "Stockholders Equity")
    goodwill_t = hist_series(bal, "Goodwill")
    other_nca_t = hist_series(bal, "Other Non Current Assets")
    oncl_t = hist_series(bal, "Other Non Current Liabilities")
    cur_lease_t = hist_series(bal, "Current Capital Lease Obligation")
    lt_lease_t = hist_series(bal, "Long Term Capital Lease Obligation")
    # Debt targets
    st_debt_t = hist_series(bal, "Current Debt")  # current debt / other current borrowings
    lt_debt_t = hist_series(bal, "Long Term Debt")  # long term debt (excluding leases)
    taxpay_t = hist_series(bal, "Total Tax Payable")
    # AOCI total from balance sheet: Gains Losses Not Affecting Retained Earnings
    aoci_t = hist_series(bal, "Gains Losses Not Affecting Retained Earnings")

    loss_bs = tf.constant(0.0, dtype=tf.float32)
    for target, model in [
        (cash_t, stmts.cash),
        (ar_t, stmts.ar),
        (ap_t, stmts.ap),
        (inv_t, stmts.inventory),
        (oca_t, stmts.other_current_assets),
        (ppe_t, stmts.net_ppe),
        (eq_t, stmts.equity),
        (aoci_t, stmts.aoci),
        (goodwill_t, stmts.goodwill_intangibles),
        (other_nca_t, stmts.other_non_current_assets),
        (oncl_t, stmts.other_non_current_liabilities),
        (cur_lease_t, stmts.current_capital_lease_obligation),
        (lt_lease_t, stmts.long_term_capital_lease_obligation),
        (st_debt_t, stmts.st_debt),
        (lt_debt_t, stmts.lt_debt),
        (taxpay_t, stmts.tax_payable),
    ]:
        if target is not None:
            loss_bs += rel_rmse(model, target)

    # IS targets
    sales_t = hist_series(fin, "Total Revenue")
    gp_t = hist_series(fin, "Gross Profit")
    # Use total operating expenses as the opex target. Some SG&A rows are
    # missing in the vendor CSV, but the "Operating Expense" line is
    # populated and numerically matches total opex.
    opex_t = hist_series(fin, "Operating Expense") # or hist_series(fin, ["other", "operating", "expenses"])
    # Depreciation / amortization from financials (positive expense)
    depr_t = hist_series(fin, "Reconciled Depreciation")
    ni_t = hist_series(fin, "Net Income")

    loss_is = tf.constant(0.0, dtype=tf.float32)
    for target, model in [
        (sales_t, stmts.sales),
        (gp_t, stmts.gross_profit),
        (opex_t, stmts.opex),
        (depr_t, stmts.depreciation),
        (ni_t, stmts.net_income),
    ]:
        if target is not None:
            loss_is += rel_rmse(model, target)

    # Cash-flow level: change in working capital from CF vs model wc_change.
    # We use a *relative* RMSE, consistent with other components, but with a
    # slightly larger epsilon in the denominator to handle near-zero WC deltas.
    def wc_rel_rmse(stmts_wc: tf.Tensor, cf_df: pd.DataFrame, cols: List[str]) -> tf.Tensor:
        r = _find_row(cf_df, "Change In Working Capital")
        if r is None:
            return tf.constant(0.0, dtype=tf.float32)
        vals = cf_df.loc[r, cols].to_numpy(dtype=np.float32)
        target = tf.convert_to_tensor(vals.reshape(1, len(cols), 1), dtype=tf.float32)
        denom = tf.maximum(tf.maximum(tf.abs(target), tf.abs(stmts_wc)), 1e-3)
        gap_rel = (stmts_wc - target) / denom
        return tf.sqrt(tf.reduce_mean(tf.square(gap_rel)))

    # To keep the calibration_loss API minimal, we re-read CF here using the
    # same path logic as load_quarterlies_train; this adds a small overhead
    # but simplifies the function signature.
    cf_full = pd.read_csv(CF_PATH, index_col=0)
    common_cols = sorted(set(cf_full.columns) & set(cols))
    cf_hist = cf_full[common_cols]
    wc_t = wc_rel_rmse(stmts.wc_change, cf_hist, cols)

    # Weighted combination, with BS now directly including OCA,
    # leases, and debt lines alongside the other balance sheet
    # targets. Working-capital CF remains a separate term.
    w_identity = 0.2
    w_bs = 0.5
    w_is = 0.3
    # w_wc = 0.05

    return w_identity * loss_identity + w_bs * loss_bs + w_is * loss_is # + w_wc * wc_t


def main(steps: int = 200, lr: float = 1e-2) -> None:
    q = load_quarterlies_train()
    bal, fin, cf, cols = q["bal"], q["fin"], q["cf"], q["cols"]

    T_hist = len(cols)
    base_policies = load_initial_policies()
    theta = make_trainable_policies(base_policies, T_hist)

    prev = build_prev_state_train(bal, cols)
    drivers = build_historical_drivers(bal, fin, cf, cols)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    layer = StructuralLayer(hard_identity_check=False, identity_tol=1e4)

    # Use eager mode for the structural layer (no @tf.function) because the
    # current implementation uses .numpy() inside the call loop.
    def train_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            pol = apply_trainable_policies(base_policies, theta, T_hist)
            stmts = layer.call(drivers, pol, prev, training=True)
            loss = calibration_loss(stmts, bal, fin, cols)
        grads = tape.gradient(loss, list(theta.values()))
        opt.apply_gradients(zip(grads, list(theta.values())))
        return loss, stmts.opex

    for step in range(steps):
        loss_val, opex_val = train_step()
        loss_val = float(loss_val.numpy())
        opex_val = opex_val.numpy().flatten()
        if (step + 1) % 20 == 0 or step == 0:
            print(f"Step {step+1}/{steps} - loss={loss_val:.4e} - opex={opex_val}")

    # Build calibrated full-horizon policies by overwriting the historical
    # window of the original forward policies, keeping forecast tail intact.
    def overwrite_hist(orig: tf.Tensor, var_name: str) -> np.ndarray:
        arr = orig.numpy().copy()
        if var_name in theta:
            v = float(theta[var_name].numpy())
            arr[:, :T_hist, :] = v
        return arr

    np.savez(
        MODEL_DIR / "wmt_policies_calibrated.npz",
        inflation=base_policies.inflation.numpy(),
        real_st_rate=base_policies.real_st_rate.numpy(),
        real_lt_rate=base_policies.real_lt_rate.numpy(),
        tax_rate=overwrite_hist(base_policies.tax_rate, "tax_rate"),
        payout_ratio=overwrite_hist(base_policies.payout_ratio, "payout_ratio"),
        min_cash_ratio=base_policies.min_cash_ratio.numpy(),
        lt_share_for_capex=base_policies.lt_share_for_capex.numpy(),
        st_invest_spread=base_policies.st_invest_spread.numpy(),
        debt_spread=base_policies.debt_spread.numpy(),
        dso_days=overwrite_hist(base_policies.dso_days, "dso_days"),
        dpo_days=overwrite_hist(base_policies.dpo_days, "dpo_days"),
        dio_days=overwrite_hist(base_policies.dio_days, "dio_days"),
        opex_ratio=overwrite_hist(base_policies.opex_ratio, "opex_ratio"),
        depreciation_rate=overwrite_hist(base_policies.depreciation_rate, "depreciation_rate"),
        gross_margin=overwrite_hist(base_policies.gross_margin, "gross_margin"),
        premium_ratio_goodwill=overwrite_hist(base_policies.premium_ratio_goodwill, "premium_ratio_goodwill"),
        beta1_capex=overwrite_hist(base_policies.beta1_capex, "beta1_capex"),
        beta2_net_invest=overwrite_hist(base_policies.beta2_net_invest, "beta2_net_invest"),
        gamma_capital_stock=overwrite_hist(base_policies.gamma_capital_stock, "gamma_capital_stock"),
        # Persist calibrated/optimized OCA coefficients so forecast can use
        # the same omega_oca_* values that best fit the historical OCA line.
        omega_oca_sales=overwrite_hist(base_policies.omega_oca_sales, "omega_oca_sales"),
        omega_oca_opex=overwrite_hist(base_policies.omega_oca_opex, "omega_oca_opex"),
        # Persist calibrated ONCL coefficients so forecast uses optimized
        # sensitivities to deferred tax and other non-cash items.
        psi_oncl_deferred_tax=overwrite_hist(base_policies.psi_oncl_deferred_tax, "psi_oncl_deferred_tax"),
        psi_oncl_other_nc=overwrite_hist(base_policies.psi_oncl_other_nc, "psi_oncl_other_nc"),
        lease_addition_capex_coeff=overwrite_hist(
            base_policies.lease_addition_capex_coeff, "lease_addition_capex_coeff"
        ),
        lease_avg_remaining_term=overwrite_hist(
            base_policies.lease_avg_remaining_term, "lease_avg_remaining_term"
        ),
    )

    print("Saved calibrated WMT policies to", MODEL_DIR)


if __name__ == "__main__":
    main()
