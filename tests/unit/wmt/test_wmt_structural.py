import pytest

try:
    import tensorflow as tf
except ImportError:  # Skip all tests if TF not available
    pytest.skip("TensorFlow not installed", allow_module_level=True)

from wmt_bs_forecaster import (
    PoliciesWMT,
    DriversWMT,
    PrevStateWMT,
    StructuralLayer,
)


def _const(batch: int, steps: int, value: float) -> tf.Tensor:
    return tf.ones([batch, steps, 1], dtype=tf.float32) * value


def _prev(batch: int, value: float) -> tf.Tensor:
    return tf.ones([batch, 1], dtype=tf.float32) * value


def build_simple(batch=1, steps=4):
    drivers = DriversWMT(
        sales=_const(batch, steps, 100.0),
        cogs=_const(batch, steps, 60.0),
        capex=None,
    )
    policies = PoliciesWMT(
        inflation=_const(batch, steps, 0.0),
        real_st_rate=_const(batch, steps, 0.0),
        real_lt_rate=_const(batch, steps, 0.0),
        tax_rate=_const(batch, steps, 0.25),
        min_cash_ratio=_const(batch, steps, 0.02),
        cash_coverage=None,
        lt_share_for_capex=_const(batch, steps, 0.8),
        st_invest_spread=_const(batch, steps, -0.01),
        debt_spread=_const(batch, steps, 0.03),
        payout_ratio=_const(batch, steps, 0.4),
        dso_days=_const(batch, steps, 30.0),
        dpo_days=_const(batch, steps, 45.0),
        dio_days=_const(batch, steps, 40.0),
        opex_ratio=_const(batch, steps, 0.18),
        depreciation_rate=_const(batch, steps, 0.05),
    )
    cash0 = 10.0
    sti0 = 0.0
    st0 = 5.0
    lt0 = 20.0
    ar0 = 30.0
    ap0 = 25.0
    inv0 = 40.0
    ppe0 = 200.0
    # Balance beginning identity for prev state
    equity0 = (cash0 + sti0 + ar0 + inv0 + ppe0) - (st0 + lt0 + ap0)
    prev = PrevStateWMT(
        cash=_prev(batch, cash0),
        st_investments=_prev(batch, sti0),
        st_debt=_prev(batch, st0),
        lt_debt=_prev(batch, lt0),
        ar=_prev(batch, ar0),
        ap=_prev(batch, ap0),
        inventory=_prev(batch, inv0),
        net_ppe=_prev(batch, ppe0),
        equity=_prev(batch, equity0),
        retained_earnings=None,
        paid_in_capital=None,
    )
    return drivers, policies, prev


def test_identity_gap_small():
    drivers, policies, prev = build_simple()
    layer = StructuralLayer()
    stmts = layer.call(drivers, policies, prev)
    gap = tf.reduce_max(tf.abs(stmts.assets - stmts.liab_plus_equity))
    assert float(gap.numpy()) < 1e-4


def test_retained_earnings_roll():
    drivers, policies, prev = build_simple()
    layer = StructuralLayer()
    stmts = layer.call(drivers, policies, prev)
    retained = stmts.retained_earnings
    net_income = stmts.net_income
    dividends = stmts.dividends
    # Check incremental updates
    delta_retained = retained[:,1:,:] - retained[:,:-1,:]
    expected = net_income[:,1:,:] - dividends[:,1:,:]
    max_err = tf.reduce_max(tf.abs(delta_retained - expected))
    assert float(max_err.numpy()) < 1e-5
