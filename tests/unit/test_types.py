import tensorflow as tf
import numpy as np

from balance_sheet_forecaster.types import Statements, Drivers, Policies, PrevState

def test_all_tensors_float32(ones_bt, ones_b1):
    B,T=2,3
    s = Statements(
        sales=ones_bt(B,T,1), cogs=ones_bt(B,T,1), opex=ones_bt(B,T,1),
        ebit=ones_bt(B,T,1), interest=ones_bt(B,T,0), tax=ones_bt(B,T,0),
        net_income=ones_bt(B,T,0),
        cash=ones_bt(B,T,1), ar=ones_bt(B,T,1), ap=ones_bt(B,T,1),
        inventory=ones_bt(B,T,1), st_investments=ones_bt(B,T,1),
        st_debt=ones_bt(B,T,1), lt_debt=ones_bt(B,T,1), nfa=ones_bt(B,T,1),
        equity=ones_bt(B,T,1), ncb=ones_bt(B,T,0)
    )
    for v in s.__dict__.values():
        assert v.dtype==tf.float32

def test_statements_assets_and_lpe_match_manual_sum(ones_bt):
    B, T = 2, 3
    stm = Statements(
        sales=ones_bt(B, T, 100),
        cogs=ones_bt(B, T, 60),
        opex=ones_bt(B, T, 20),
        ebit=ones_bt(B, T, 20),
        interest=ones_bt(B, T, 1),
        tax=ones_bt(B, T, 4),
        net_income=ones_bt(B, T, 15),

        cash=ones_bt(B, T, 5),
        ar=ones_bt(B, T, 3),
        ap=ones_bt(B, T, 2),
        inventory=ones_bt(B, T, 4),
        st_investments=ones_bt(B, T, 1.5),
        st_debt=ones_bt(B, T, 1),
        lt_debt=ones_bt(B, T, 10),
        nfa=ones_bt(B, T, 20),
        equity=ones_bt(B, T, 20.5),

        ncb=ones_bt(B, T, 0),
    )

    # Manual sums
    assets_manual = stm.cash + stm.ar + stm.inventory + stm.st_investments + stm.nfa
    lpe_manual    = stm.ap + stm.st_debt + stm.lt_debt + stm.equity

    # Property outputs equal manual sums and have correct shape
    assert stm.assets.shape == (B, T, 1)
    assert stm.liab_plus_equity.shape == (B, T, 1)
    assert np.allclose(stm.assets.numpy(), assets_manual.numpy(), atol=1e-7)
    assert np.allclose(stm.liab_plus_equity.numpy(), lpe_manual.numpy(), atol=1e-7)

def test_balance_sheet_view_shapes(ones_bt):
    B,T=1,2
    s = Statements(
        sales=ones_bt(B,T,0), cogs=ones_bt(B,T,0), opex=ones_bt(B,T,0),
        ebit=ones_bt(B,T,0), interest=ones_bt(B,T,0), tax=ones_bt(B,T,0),
        net_income=ones_bt(B,T,0),
        cash=ones_bt(B,T,5), ar=ones_bt(B,T,1), ap=ones_bt(B,T,2),
        inventory=ones_bt(B,T,3), st_investments=ones_bt(B,T,4),
        st_debt=ones_bt(B,T,6), lt_debt=ones_bt(B,T,7), nfa=ones_bt(B,T,8),
        equity=ones_bt(B,T,9), ncb=ones_bt(B,T,0)
    )
    view=s.balance_sheet_view()
    for k,v in view.items():
        assert v.shape==(B,T,1), f"{k} wrong shape"

def test_balance_sheet_view_keys_and_values(ones_bt):
    B, T = 1, 2
    stm = Statements(
        sales=ones_bt(B, T, 0),
        cogs=ones_bt(B, T, 0),
        opex=ones_bt(B, T, 0),
        ebit=ones_bt(B, T, 0),
        interest=ones_bt(B, T, 0),
        tax=ones_bt(B, T, 0),
        net_income=ones_bt(B, T, 0),

        cash=ones_bt(B, T, 5),
        ar=ones_bt(B, T, 1),
        ap=ones_bt(B, T, 2),
        inventory=ones_bt(B, T, 3),
        st_investments=ones_bt(B, T, 4),
        st_debt=ones_bt(B, T, 6),
        lt_debt=ones_bt(B, T, 7),
        nfa=ones_bt(B, T, 8),
        equity=ones_bt(B, T, 9),

        ncb=ones_bt(B, T, 0),
    )

    view = stm.balance_sheet_view()
    # Expected keys exist
    expected = {
        "cash", "short_term_investments", "accounts_receivable", "accounts_payable",
        "inventory", "short_term_debt", "long_term_debt", "net_fixed_assets",
        "equity", "total_assets", "total_liabilities_and_equity",
    }
    assert expected.issubset(view.keys())

    # Values match underlying fields (spot checks)
    assert np.allclose(view["cash"].numpy(), stm.cash.numpy())
    assert np.allclose(view["accounts_payable"].numpy(), stm.ap.numpy())
    assert np.allclose(view["total_assets"].numpy(), stm.assets.numpy())
    assert np.allclose(view["total_liabilities_and_equity"].numpy(), stm.liab_plus_equity.numpy())

def test_drivers_construction_and_shapes(ones_bt):
    B, T = 2, 4
    drv = Drivers(
        price=ones_bt(B, T, 10),
        volume=ones_bt(B, T, 2),
        dso_days=ones_bt(B, T, 30),
        dpo_days=ones_bt(B, T, 25),
        dio_days=ones_bt(B, T, 35),
        capex=ones_bt(B, T, 1),
        stlt_split=ones_bt(B, T, 0.4),
    )
    # All fields exist and have [B,T,1]
    for v in drv.__dict__.values():
        assert isinstance(v, tf.Tensor)
        assert v.shape == (B, T, 1)
        assert v.dtype == tf.float32

def test_policies_required_and_optional_fields(ones_bt):
    B, T = 1, 3
    pol = Policies(
        inflation=ones_bt(B, T, 0.02),
        real_rate=ones_bt(B, T, 0.01),
        tax_rate=ones_bt(B, T, 0.25),
        min_cash_ratio=ones_bt(B, T, 0.05),
        payout_ratio=ones_bt(B, T, 0.2),

        # Leave optionals as None by default
    )
    # Required shapes
    for name in ["inflation", "real_rate", "tax_rate", "min_cash_ratio", "payout_ratio"]:
        t = getattr(pol, name)
        assert t.shape == (B, T, 1)
        assert t.dtype == tf.float32

    # Optionals default None
    for name in ["lt_rate", "opex_ratio", "depreciation_rate", "cost_share", "st_rate", "st_invest_rate", "cash_coverage"]:
        assert getattr(pol, name) is None

    # Setting an optional should take effect with correct shape
    pol2 = Policies(
        inflation=ones_bt(B, T, 0.02),
        real_rate=ones_bt(B, T, 0.01),
        tax_rate=ones_bt(B, T, 0.25),
        min_cash_ratio=ones_bt(B, T, 0.05),
        payout_ratio=ones_bt(B, T, 0.2),
        lt_rate=ones_bt(B, T, 0.06),
    )
    assert pol2.lt_rate.shape == (B, T, 1)

def test_policies_optional_passthrough(ones_bt):
    B,T=1,2
    p = Policies(
        inflation=ones_bt(B,T,0.02),
        real_rate=ones_bt(B,T,0.01),
        tax_rate=ones_bt(B,T,0.25),
        min_cash_ratio=ones_bt(B,T,0.05),
        payout_ratio=ones_bt(B,T,0.20),
        lt_rate=ones_bt(B,T,0.06),
    )
    assert p.lt_rate.shape==(B,T,1)
    assert p.st_rate is None and p.cash_coverage is None

def test_prevstate_shapes(ones_b1):
    B = 3
    prev = PrevState(
        cash=ones_b1(B, 5),
        st_investments=ones_b1(B, 0),
        st_debt=ones_b1(B, 2),
        lt_debt=ones_b1(B, 10),
        ar=ones_b1(B, 3),
        ap=ones_b1(B, 1),
        inventory=ones_b1(B, 4),
        nfa=ones_b1(B, 20),
        equity=ones_b1(B, 19),
    )
    for v in prev.__dict__.values():
        assert isinstance(v, tf.Tensor)
        assert v.shape == (B, 1)
        assert v.dtype == tf.float32

def test_statements_handles_negatives_and_nans(ones_bt):
    B,T=1,1
    s = Statements(
        sales=ones_bt(B,T,-1), cogs=ones_bt(B,T, np.nan), opex=ones_bt(B,T,0),
        ebit=ones_bt(B,T,0), interest=ones_bt(B,T,0), tax=ones_bt(B,T,0),
        net_income=ones_bt(B,T,0),
        cash=ones_bt(B,T,0), ar=ones_bt(B,T,0), ap=ones_bt(B,T,0),
        inventory=ones_bt(B,T,0), st_investments=ones_bt(B,T,0),
        st_debt=ones_bt(B,T,0), lt_debt=ones_bt(B,T,0), nfa=ones_bt(B,T,0),
        equity=ones_bt(B,T,0), ncb=ones_bt(B,T,0)
    )
    _ = s.assets  # should compute without raising
    _ = s.liab_plus_equity
