import numpy as np
import tensorflow as tf

from wmt_bs_forecaster.types_wmt import (
    PoliciesWMT,
    DriversWMT,
    PrevStateWMT,
    StatementsWMT,
)


def _const_series(value: float, T: int = 3, B: int = 1) -> tf.Tensor:
    return tf.ones([B, T, 1], dtype=tf.float32) * value


def _prev_scalar(value: float, B: int = 1) -> tf.Tensor:
    return tf.ones([B, 1], dtype=tf.float32) * value


def test_policieswmt_minimal_shapes():
    B, T = 1, 4
    pol = PoliciesWMT(
        inflation=_const_series(0.0, T, B),
        real_st_rate=_const_series(0.01, T, B),
        real_lt_rate=_const_series(0.02, T, B),
        tax_rate=_const_series(0.25, T, B),
        payout_ratio=_const_series(0.4, T, B),
        min_cash_ratio=_const_series(0.02, T, B),
        lt_share_for_capex=_const_series(0.6, T, B),
        st_invest_spread=_const_series(0.001, T, B),
        debt_spread=_const_series(0.02, T, B),
        dso_days=_const_series(30.0, T, B),
        dpo_days=_const_series(40.0, T, B),
        dio_days=_const_series(35.0, T, B),
        opex_ratio=_const_series(0.18, T, B),
        depreciation_rate=_const_series(0.05, T, B),
        period_days=365.0 / 4.0,
    )

    assert pol.inflation.shape == (B, T, 1)
    assert pol.dso_days.shape == (B, T, 1)
    assert pol.depreciation_rate.shape == (B, T, 1)
    # Optional fields should default to None or sensible tensors
    assert getattr(pol, "gross_margin", None) is None or isinstance(pol.gross_margin, tf.Tensor)


def test_driverswmt_basic_construction():
    B, T = 2, 5
    sales = _const_series(100.0, T, B)
    cogs = _const_series(60.0, T, B)
    drv = DriversWMT(
        sales=sales,
        cogs=cogs,
        capex=None,
    )
    assert drv.sales.shape == (B, T, 1)
    assert drv.cogs.shape == (B, T, 1)
    assert drv.capex is None


def test_prevstatewmt_identity_consistency_small_example():
    B = 1
    cash = 10.0
    sti = 5.0
    ar = 20.0
    inv = 15.0
    ppe = 50.0
    st_debt = 8.0
    lt_debt = 25.0
    ap = 12.0
    # Equity chosen to balance
    equity = (cash + sti + ar + inv + ppe) - (st_debt + lt_debt + ap)

    prev = PrevStateWMT(
        cash=_prev_scalar(cash, B),
        st_investments=_prev_scalar(sti, B),
        st_debt=_prev_scalar(st_debt, B),
        lt_debt=_prev_scalar(lt_debt, B),
        ar=_prev_scalar(ar, B),
        ap=_prev_scalar(ap, B),
        inventory=_prev_scalar(inv, B),
        net_ppe=_prev_scalar(ppe, B),
        equity=_prev_scalar(equity, B),
    )

    # Spot‑check values
    assert float(prev.cash.numpy()[0, 0]) == cash
    assert float(prev.equity.numpy()[0, 0]) == equity


def test_statementswmt_assets_and_liab_plus_equity_consistent_definition():
    # Construct a one‑step synthetic statement and verify the identity helper
    B, T = 1, 1
    def s(x):
        return tf.reshape(tf.constant([[x]], dtype=tf.float32), [B, T, 1])

    cash = 10.0
    sti = 2.0
    ar = 5.0
    inv = 7.0
    other_cur = 3.0
    ppe = 20.0
    goodwill = 4.0
    other_nca = 6.0

    st_debt = 8.0
    cur_lease = 1.0
    lt_debt = 15.0
    lt_lease = 2.0
    ap = 5.0
    accr = 3.0
    taxp = 1.0
    divp = 2.0
    oncl = 4.0
    equity = 18.0
    cap_stock = 1.0
    aoci = -1.0
    minority = 4.0

    stm = StatementsWMT(
        sales=s(0.0), cogs=s(0.0), gross_profit=s(0.0), opex=s(0.0), ebit=s(0.0),
        interest_income=s(0.0), interest_expense=s(0.0), ebt=s(0.0), taxes=s(0.0), net_income=s(0.0),
        capex=s(0.0), depreciation=s(0.0), wc_change=s(0.0),
        cash=s(cash), st_investments=s(sti), st_debt=s(st_debt), lt_debt=s(lt_debt),
        ar=s(ar), ap=s(ap), inventory=s(inv), net_ppe=s(ppe), equity=s(equity),
        dividends=s(0.0), retained_earnings=s(0.0), paid_in_capital=s(0.0),
        other_current_assets=s(other_cur), goodwill_intangibles=s(goodwill), other_non_current_assets=s(other_nca),
        accrued_expenses=s(accr), tax_payable=s(taxp), other_non_current_liabilities=s(oncl),
        aoci=s(aoci), minority_interest=s(minority),
        current_capital_lease_obligation=s(cur_lease), long_term_capital_lease_obligation=s(lt_lease),
        dividends_payable=s(divp), capital_stock=s(cap_stock),
    )

    assets = stm.assets.numpy()[0, 0, 0]
    liab_eq = stm.liab_plus_equity.numpy()[0, 0, 0]

    # Compute the same numbers manually to ensure consistency with helper properties
    assets_manual = cash + sti + ar + inv + other_cur + ppe + goodwill + other_nca
    liab_eq_manual = (
        st_debt + cur_lease + lt_debt + lt_lease + ap + accr + taxp + divp + oncl
        + (equity + cap_stock + aoci) + minority
    )

    assert np.isclose(assets, assets_manual)
    assert np.isclose(liab_eq, liab_eq_manual)
