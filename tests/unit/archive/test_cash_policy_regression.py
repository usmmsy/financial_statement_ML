import numpy as np
import tensorflow as tf

from balance_sheet_forecaster.accounting import (
    StructuralLayer,
    fisher_nominal, 
    clamp_positive, 
    days_to_balance, 
)
from balance_sheet_forecaster.types import Drivers, Policies, PrevState


# ---------- regression tests ----------

def test_cash_policy_deficit_borrows_and_hits_floor(struct, mk_drivers, mk_policies, mk_prev, scalar, ones_bt):
    """
    If pre_cash < min_cash, we must:
      - borrow exactly the deficit
      - split ST/LT by stlt_split
      - set st_investments to 0
      - keep cash = pre_cash + borrowed (>= min_cash)
      - interest uses ONLY previous balances (no circularity)
    """
    B, T = 1, 1
    stlt_split = 0.40

    # Sales=10, min_cash_ratio=0.2  -> min_cash = 2.0
    drv = mk_drivers(B, T, price=10.0, volume=1.0, capex=0.8, stlt_split=stlt_split)  # cash_out includes capex
    pol = mk_policies(B, T, min_cash_ratio=0.20, payout_ratio=0.0)
    prev = mk_prev(B, cash=0.0, st_inv=0.0, st_debt=1.0, lt_debt=3.0, nfa=0.0, eq=5.0)

    stm = struct(drivers=drv, policies=pol, prev=prev, training=False)

    sales = 10.0
    min_cash = pol.min_cash_ratio * sales  # policy floor
    # End cash equals pre_cash + new borrowing and reaches (or exceeds numerically) the floor
    assert np.isclose(scalar(stm.cash), min_cash, atol=1e-5)
    # st_investments must be zero in deficit case
    assert np.isclose(scalar(stm.st_investments), 0.0, atol=1e-6)

    # Total new debt is the deficit; infer it from deltas
    delta_st = scalar(stm.st_debt) - scalar(prev.st_debt)
    delta_lt = scalar(stm.lt_debt) - scalar(prev.lt_debt)
    borrowed = max(delta_st + delta_lt, 0.0)
    assert borrowed > 0.0
    
    # Split matches stlt_split
    # (Allow small tolerance for floating/eps in internals.)
    assert np.isclose(delta_lt, stlt_split * borrowed, atol=1e-4)
    assert np.isclose(delta_st, (1.0 - stlt_split) * borrowed, atol=1e-4)

    # No circularity: interest depends only on previous balances
    prev_hi = mk_prev(B, cash=0.0, st_inv=0.0, st_debt=2.0, lt_debt=3.0, nfa=0.0, eq=5.0)
    stm_hi = struct(drivers=drv, policies=pol, prev=prev_hi, training=False)
    hi_interest = scalar(stm_hi.interest[:, 0, 0])
    base_interest = scalar(stm.interest[:, 0, 0])
    assert hi_interest > base_interest

    # expected interest difference from extra 1.0 ST debt should match nominal short rate
    expected_st_rate = fisher_nominal(ones_bt(1, 1, pol.real_rate), ones_bt(1, 1, pol.inflation))
    assert np.isclose(hi_interest - base_interest, expected_st_rate, atol=1e-6)  # only the extra 1.0 ST debt should contribute


def test_cash_policy_excess_parks_in_st_investments(struct, mk_drivers, mk_policies, mk_prev, scalar, ones_bt):
    """
    If pre_cash > min_cash, we must:
      - NOT borrow new debt
      - leave cash at pre_cash (min cash is a floor, not a target)
      - send (pre_cash - min_cash) to st_investments
    """
    B, T = 1, 1

    # Keep things simple so cash_in - cash_out is positive and well above the floor.
    drv = mk_drivers(B, T, price=10.0, volume=1.0, capex=0.0)
    pol = mk_policies(B, T, min_cash_ratio=0.05, payout_ratio=0.0, inflation=0.0, real=0.0, tax=0.0)
    prev = mk_prev(B, cash=0.0, st_inv=0.0, st_debt=0.0, lt_debt=0.0, nfa=0.0, eq=5.0)

    stm = struct(drivers=drv, policies=pol, prev=prev, training=False)

    sales = 10.0
    min_cash = pol.min_cash_ratio * sales  # 0.5
    
    # With a surplus, debt deltas should be ~0
    assert np.isclose(scalar(stm.st_debt), scalar(prev.st_debt), atol=1e-6)
    assert np.isclose(scalar(stm.lt_debt), scalar(prev.lt_debt), atol=1e-6)

    # Surplus path: cash is above the floor and st_investments = cash - min_cash
    cash_val = scalar(stm.cash)
    assert cash_val >= min_cash - 1e-6
    assert np.isclose(scalar(stm.st_investments), max(cash_val - min_cash, 0.0), atol=1e-5)

