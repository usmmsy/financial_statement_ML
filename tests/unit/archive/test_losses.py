import numpy as np
import tensorflow as tf
import pytest

from balance_sheet_forecaster.losses import (
    mae,
    statement_fit_loss,
    identity_guardrail,
    smoothness_penalty,
)
from balance_sheet_forecaster.types import Statements, Drivers


def _bt(B, T, v):
    return tf.ones([B, T, 1], tf.float32) * float(v)


def _zeros(B, T):
    return tf.zeros([B, T, 1], tf.float32)


def _stm_all_zeros(B, T):
    # Build a Statements object with all lines = 0
    z = _zeros(B, T)
    return Statements(
        sales=z, cogs=z, opex=z, ebit=z, interest=z, tax=z, net_income=z,
        cash=z, ar=z, ap=z, inventory=z, st_investments=z, st_debt=z,
        lt_debt=z, nfa=z, equity=z, ncb=z
    )


def _drivers_constant(B, T, v=0.0):
    c = _bt(B, T, v)
    return Drivers(
        price=c, volume=c, dso_days=c, dpo_days=c, dio_days=c, capex=c, stlt_split=c
    )


# -------------------- mae --------------------

def test_mae_basic():
    x = tf.constant([[[ -2.0 ]], [[ 0.0 ]], [[ 3.0 ]]], dtype=tf.float32)  # [3,1,1]
    out = mae(x)
    # Mean(|-2|, |0|, |3|) = (2 + 0 + 3) / 3 = 5/3
    assert np.isclose(float(out.numpy()), 5.0/3.0, rtol=1e-6, atol=1e-6)


# -------------------- statement_fit_loss --------------------

def test_statement_fit_loss_weights_and_skip():
    B, T = 2, 3

    # Pred statements: sales slightly off vs true; cash exact vs true.
    pred = _stm_all_zeros(B, T)
    pred = Statements(
        **{**pred.__dict__,
           "sales": _bt(B, T, 11.0),  # true will be 10
           "cash":  _bt(B, T, 5.0)}   # true will be 5
    )

    # True targets provide only 'sales' and 'cash' (others should be skipped).
    true = {
        "sales": _bt(B, T, 10.0),
        "cash":  _bt(B, T, 5.0)
    }

    # Weights: give 'sales' a weight of 2.0, cash defaults to 1.0
    weights = {"sales": 2.0}

    # L1 diff: |11-10| = 1 everywhere -> mean = 1
    # 'cash' diff = 0 -> mean 0
    # Weighted sum = 2.0 * 1 + 1.0 * 0 = 2.0
    loss = statement_fit_loss(pred, true, weights)
    assert np.isclose(float(loss.numpy()), 2.0, rtol=1e-6, atol=1e-6)


def test_statement_fit_loss_handles_no_keys_gracefully():
    B, T = 1, 2
    pred = _stm_all_zeros(B, T)
    true = {}  # nothing provided
    loss = statement_fit_loss(pred, true, weights=None)
    assert np.isclose(float(loss.numpy()), 0.0, atol=1e-12)


# -------------------- identity_guardrail --------------------

def test_identity_guardrail_zero_when_identity_holds():
    B, T = 2, 3
    # Start with zeros -> assets == liab_plus_equity == 0
    pred = _stm_all_zeros(B, T)
    guard = identity_guardrail(pred)
    assert np.isclose(float(guard.numpy()), 0.0, atol=1e-12)


def test_identity_guardrail_matches_known_gap():
    B, T = 2, 4
    pred = _stm_all_zeros(B, T)

    # Create a +1 unit gap by bumping cash by +1 everywhere.
    pred = Statements(
        **{**pred.__dict__, "cash": _bt(B, T, 1.0)}
    )
    # assets - (liab+equity) = +1 everywhere -> mean abs gap = 1
    guard = identity_guardrail(pred)
    assert np.isclose(float(guard.numpy()), 1.0, rtol=1e-6, atol=1e-6)


# -------------------- smoothness_penalty --------------------

def test_smoothness_penalty_zero_for_constant_series():
    B, T = 3, 5
    drivers = _drivers_constant(B, T, v=7.0)  # all series flat in time
    pen = smoothness_penalty(drivers, lam=1.0)
    assert np.isclose(float(pen.numpy()), 0.0, atol=1e-12)


def test_smoothness_penalty_single_series_change_matches_expectation():
    B, T = 1, 3
    lam = 1.7

    # All constant except 'price', which follows [1, 4, 1]
    d = _drivers_constant(B, T, v=0.0)
    price = tf.constant([[[1.0], [4.0], [1.0]]], dtype=tf.float32)  # [1,3,1]
    d = Drivers(
        price=price,               # non-constant
        volume=d.volume,
        dso_days=d.dso_days,
        dpo_days=d.dpo_days,
        dio_days=d.dio_days,
        capex=d.capex,
        stlt_split=d.stlt_split,
    )

    # diff(price) = [3, -3] -> |.| = [3,3], mean = 3
    expected = lam * 3.0
    pen = smoothness_penalty(d, lam=lam)
    assert np.isclose(float(pen.numpy()), expected, rtol=1e-6, atol=1e-6)


def test_smoothness_penalty_scales_with_lam():
    B, T = 1, 4

    # Make two drivers vary identically to ensure additivity over series
    seq = tf.constant([[[0.0], [2.0], [2.0], [6.0]]], dtype=tf.float32)  # diffs: [2,0,4] -> mean = 2
    d = Drivers(
        price=seq,
        volume=seq,
        dso_days=_bt(B, T, 0.0),
        dpo_days=_bt(B, T, 0.0),
        dio_days=_bt(B, T, 0.0),
        capex=_bt(B, T, 0.0),
        stlt_split=_bt(B, T, 0.0),
    )

    # Two changing series -> total base = 2 (mean diff) + 2 = 4
    base = 4.0
    for lam in [0.1, 1.0, 3.5]:
        pen = smoothness_penalty(d, lam=lam)
        assert np.isclose(float(pen.numpy()), lam * base, rtol=1e-6, atol=1e-6)
