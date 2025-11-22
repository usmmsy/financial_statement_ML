import os
import pytest

try:
    import tensorflow as tf
except ImportError:
    pytest.skip("TensorFlow not installed", allow_module_level=True)

from wmt_bs_forecaster.data_wmt import load_wmt_csvs
from wmt_bs_forecaster.types_wmt import DriversWMT, PrevStateWMT
from wmt_bs_forecaster.accounting_wmt import StructuralLayer


def test_identity_real_wmt_quarterly():
    base = os.path.join("data", "retail_csv", "WMT_quarterly")
    fin = os.path.join(base, "WMT_quarterly_financials.csv")
    bal = os.path.join(base, "WMT_quarterly_balance_sheet.csv")
    cf = os.path.join(base, "WMT_quarterly_cash_flow.csv")

    # Load real data for a deterministic run
    drivers, policies, prev_state, _targets = load_wmt_csvs(
        financials_csv=fin,
        balance_csv=bal,
        cashflow_csv=cf,
        horizon=8,
        infer_wc_days=True,
    )

    # Sanitize any NaNs in drivers to avoid NaN propagation in deterministic check
    def nz(x: tf.Tensor, fill: float = 0.0) -> tf.Tensor:
        return tf.where(tf.math.is_nan(x), tf.ones_like(x) * fill, x)

    drivers = DriversWMT(
        sales=nz(drivers.sales, 0.0),
        cogs=nz(drivers.cogs, 0.0),
        capex=nz(drivers.capex, 0.0) if drivers.capex is not None else None,
    )
    # Sanitize policy schedules/ratios
    policies.dso_days = nz(policies.dso_days, 0.0)
    policies.dpo_days = nz(policies.dpo_days, 0.0)
    policies.dio_days = nz(policies.dio_days, 0.0)
    policies.opex_ratio = nz(policies.opex_ratio, 0.0)
    policies.depreciation_rate = nz(policies.depreciation_rate, 0.02)

    # Rebalance initial equity to our modeled items to avoid legacy sheet mismatches
    cash0 = prev_state.cash
    sti0 = prev_state.st_investments
    st0 = prev_state.st_debt
    lt0 = prev_state.lt_debt
    ar0 = prev_state.ar
    ap0 = prev_state.ap
    inv0 = prev_state.inventory
    ppe0 = prev_state.net_ppe
    equity0 = (cash0 + sti0 + ar0 + inv0 + ppe0) - (st0 + lt0 + ap0)
    prev_state = PrevStateWMT(
        cash=cash0,
        st_investments=sti0,
        st_debt=st0,
        lt_debt=lt0,
        ar=ar0,
        ap=ap0,
        inventory=inv0,
        net_ppe=ppe0,
        equity=equity0,
        retained_earnings=None,
        paid_in_capital=None,
    )

    layer = StructuralLayer(hard_identity_check=False, identity_tol=1e-4)
    stm = layer.call(drivers, policies, prev_state, training=False)

    # Expanded identity over full balance sheet (preferred)
    gap = tf.reduce_max(tf.abs(stm.assets - stm.liab_plus_equity))
    max_assets = tf.reduce_max(tf.abs(stm.assets))
    rel_gap = float((gap / tf.maximum(max_assets, 1.0)).numpy())
    assert rel_gap < 1e-2, f"Expanded identity gap too large: rel={rel_gap}"
