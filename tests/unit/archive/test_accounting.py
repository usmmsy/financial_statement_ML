import numpy as np
import tensorflow as tf

from balance_sheet_forecaster.accounting import (
    StructuralLayer,
    fisher_nominal, 
    clamp_positive, 
    days_to_balance, 
)
from balance_sheet_forecaster.types import Statements, Drivers, Policies, PrevState


# helper function tests
def test_fisher_nominal_exact(ones_bt):
    # (1+0.01)*(1+0.02)-1 = 0.0302
    out = fisher_nominal(ones_bt(1, 1, 0.01), ones_bt(1, 1, 0.02))
    assert tf.reduce_all(tf.abs(out - 0.0302) < 1e-7)

def test_clamp_positive_nonnegative_and_floor():
    x = tf.constant([[[-3.0]], [[0.0]], [[5.0]]], dtype=tf.float32)  # [3,1,1]
    y = clamp_positive(x, epsilon=1e-6)
    assert tf.reduce_min(y) >= 0.0
    # zero should become epsilon (strictly positive for numeric safety)
    assert tf.abs(y[1, 0, 0] - 1e-6) < 1e-12

def test_days_to_balance_linear_scaling(ones_bt):
    flow = ones_bt(1, 1, 100.0)   # per-period flow
    days = ones_bt(1, 1, 36.5)
    # balance ~= 100 * (36.5/365) = 10
    bal = days_to_balance(flow, days)
    assert tf.reduce_all(tf.abs(bal - 10.0) < 1e-6)


# StructuralLayer tests

# Tests for StructuralLayer behavior
def test_accounting_identity_holds_small_tolerance(struct, mk_drivers, mk_policies, mk_prev):
    B, T = 2, 3
    drv = mk_drivers(B, T)
    pol = mk_policies(B, T)
    prev = mk_prev(B)

    stm = struct(drivers=drv, policies=pol, prev=prev, training=False)

    # check per-period identity (not just mean)
    diff = tf.abs(stm.assets - stm.liab_plus_equity)
    assert float(tf.reduce_max(diff).numpy()) < 1e-4

# Test that interest expense depends only on prior balances, not circularly
def test_interest_uses_prev_balances_only_no_circularity(struct, mk_drivers, mk_policies, mk_prev):
    # same drivers/policies; vary previous ST debt and see interest change at t=0
    B, T = 1, 2
    drv = mk_drivers(B, T, price=1.0, volume=1.0, capex=0.0)  # keep flows simple
    pol = mk_policies(B, T, inflation=0.0, real=0.05, tax=0.0)  # positive rate, no tax

    prev_low = mk_prev(B, st_debt=5.0, lt_debt=0.0, cash=0.0, st_inv=0.0,
                       ar=0.0, ap=0.0, inv=0.0, nfa=0.0, eq=0.0)
    prev_high = mk_prev(B, st_debt=10.0, lt_debt=0.0, cash=0.0, st_inv=0.0,
                        ar=0.0, ap=0.0, inv=0.0, nfa=0.0, eq=0.0)

    interest_low = struct(drivers=drv, policies=pol, prev=prev_low, training=False).interest[:, 0, 0].numpy()
    interest_high = struct(drivers=drv, policies=pol, prev=prev_high, training=False).interest[:, 0, 0].numpy()

    assert interest_high > interest_low

# Test that min_cash_ratio policy is enforced via borrowing/ investing
def test_min_cash_policy_borrows_when_below_floor(struct, mk_drivers, mk_policies, mk_prev):
    # Force deficit: make capex big so pre_cash < min_cash
    B, T = 1, 1
    drv = mk_drivers(B, T, price=10.0, volume=1.0, capex=5.0)  # sales=10, heavy capex
    pol = mk_policies(B, T, min_cash_ratio=0.20)               # floor = 2.0
    prev = mk_prev(B, cash=0.0, st_inv=0.0, st_debt=0.0, lt_debt=0.0)

    stm = struct(drivers=drv, policies=pol, prev=prev, training=False)

    # End cash should equal the policy floor when pre_cash is below it
    assert np.isclose(stm.cash.numpy()[0,0,0], 2.0, atol=1e-6)
    # Deficit is financed: st/lt debt should increase
    assert stm.st_debt.numpy()[0,0,0] + stm.lt_debt.numpy()[0,0,0] > 0.0

# Test that min_cash_ratio policy does not borrow when above floor
def test_min_cash_policy_does_not_borrow_when_above_floor(struct, mk_drivers, mk_policies, mk_prev):
    # Force surplus: low costs; pre_cash >= min_cash
    B, T = 1, 1
    drv = mk_drivers(B, T, price=10.0, volume=1.0, capex=0.0)   # sales=10, no capex
    pol = mk_policies(B, T, min_cash_ratio=0.01)                # floor = 0.1
    prev = mk_prev(B, cash=0.0, st_inv=0.0, st_debt=0.0, lt_debt=0.0)

    stm = struct(drivers=drv, policies=pol, prev=prev, training=False)

    # End cash should follow pre_cash path (above the low floor)
    assert stm.cash.numpy()[0,0,0] >= 0.1 - 1e-6
    # No borrowing required; debt should remain ~0 (within numeric tolerance)
    assert np.isclose(stm.st_debt.numpy()[0,0,0], 0.0, atol=1e-6)
    assert np.isclose(stm.lt_debt.numpy()[0,0,0], 0.0, atol=1e-6)



# Test that inventory and COGS are coherent with DIO driver

def test_inventory_cogs_coherence_with_dio(struct, mk_drivers, mk_policies, mk_prev):
    # Given the current placeholder:
    # implied_inventory = days_to_balance(sales, dio) * 0.6
    # cogs ≈ implied_inventory * 365 / dio (then clamped positive)
    B, T = 1, 2
    sales_v = 100.0
    dio_v = 30.0

    drv = mk_drivers(B, T, price=sales_v, volume=1.0, dio=dio_v, dso=30.0, dpo=30.0, capex=0.0)
    pol = mk_policies(B, T)
    prev = mk_prev(B)

    stm = struct(drivers=drv, policies=pol, prev=prev, training=False)

    implied_inventory = days_to_balance(
        tf.ones([B, T, 1], tf.float32) * sales_v,
        tf.ones([B, T, 1], tf.float32) * dio_v
    ) * 0.6
    cogs_expected = tf.nn.relu(
        implied_inventory * 365.0 / (tf.ones([B, T, 1], tf.float32) * dio_v + 1e-6)
    )

    mae = tf.reduce_mean(tf.abs(stm.cogs - cogs_expected)).numpy()
    assert mae < 1e-5

