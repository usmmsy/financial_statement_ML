import os
import random

import numpy as np
import pytest
import tensorflow as tf

from balance_sheet_forecaster.accounting import StructuralLayer
from balance_sheet_forecaster.types import Drivers, Policies, PrevState

# --- Global test setup (runs once per test session) ---------------------------
def pytest_sessionstart(session):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=warn, 2=error


# --- Determinism --------------------------------------------------------------
@pytest.fixture(autouse=True)
def set_seeds():
    # Keep per-test determinism
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    yield

# --- Tiny tensor helpers ------------------------------------------------------
@pytest.fixture
def ones_bt():
    def _ones_bt(B, T, v):
        return tf.ones([B, T, 1], tf.float32) * float(v)
    return _ones_bt


@pytest.fixture
def ones_b1():
    def _ones_b1(B, v):
        return tf.ones([B, 1], tf.float32) * float(v)
    return _ones_b1


@pytest.fixture
def scalar():
    def _scalar(x):
        return float(x.numpy().reshape(-1)[0])
    return _scalar

# --- Tiny tensor builders -----------------------------------------------------
@pytest.fixture
def mk_drivers():
    """
    Factory returning a function that builds a Drivers instance with defaults.
    Usage in tests:
        drv = mk_drivers(B, T, price=10.0, volume=1.0, ...)
    """
    def _mk_drivers(B, T, *, price=10.0, volume=1.0,
                    dso=30.0, dpo=30.0, dio=30.0,
                    capex=0.0, stlt_split=0.4):
        ones = lambda v: tf.ones([B, T, 1], tf.float32) * float(v)
        return Drivers(
            price=ones(price),
            volume=ones(volume),
            dso_days=ones(dso),
            dpo_days=ones(dpo),
            dio_days=ones(dio),
            capex=ones(capex),
            stlt_split=ones(stlt_split),
        )
    return _mk_drivers


@pytest.fixture
def mk_policies():
    """
    Factory returning a function that builds a Policies instance with defaults.
    Any argument set to None will be left as None (so your layer defaults kick in).
    """
    def _mk_policies(B, T, *, inflation=0.02, real=0.01, tax=0.25,
                     min_cash_ratio=0.10, payout_ratio=0.0,
                     lt_rate=None, opex_ratio=None, depreciation_rate=None,
                     cost_share=None, st_rate=None, st_invest_rate=None,
                     cash_coverage=None):
        def maybe(v):
            return None if v is None else tf.ones([B, T, 1], tf.float32) * float(v)
        return Policies(
            inflation=tf.ones([B, T, 1], tf.float32) * float(inflation),
            real_rate=tf.ones([B, T, 1], tf.float32) * float(real),
            tax_rate=tf.ones([B, T, 1], tf.float32) * float(tax),
            min_cash_ratio=tf.ones([B, T, 1], tf.float32) * float(min_cash_ratio),
            payout_ratio=tf.ones([B, T, 1], tf.float32) * float(payout_ratio),
            lt_rate=maybe(lt_rate),
            opex_ratio=maybe(opex_ratio),
            depreciation_rate=maybe(depreciation_rate),
            cost_share=maybe(cost_share),
            st_rate=maybe(st_rate),
            st_invest_rate=maybe(st_invest_rate),
            cash_coverage=maybe(cash_coverage),
        )
    return _mk_policies


@pytest.fixture
def mk_prev():
    """
    Factory returning a function that builds a PrevState instance.
    """
    def _mk_prev(B, *, cash=0.0, st_inv=0.0, st_debt=0.0, lt_debt=0.0,
                 ar=0.0, ap=0.0, inv=0.0, nfa=0.0, eq=0.0):
        to_b1 = lambda v: tf.ones([B, 1], tf.float32) * float(v)
        return PrevState(
            cash=to_b1(cash),
            st_investments=to_b1(st_inv),
            st_debt=to_b1(st_debt),
            lt_debt=to_b1(lt_debt),
            ar=to_b1(ar),
            ap=to_b1(ap),
            inventory=to_b1(inv),
            nfa=to_b1(nfa),
            equity=to_b1(eq),
        )
    return _mk_prev


# --- Structural layer fixture -------------------------------------------------
@pytest.fixture
def struct():
    """Fresh StructuralLayer per test."""
    return StructuralLayer()
