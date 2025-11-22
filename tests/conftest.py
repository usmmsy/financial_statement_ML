import os
import random

import numpy as np
import pytest
import tensorflow as tf

import sys
import pathlib

# Ensure project src/ is on sys.path so wmt_bs_forecaster is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wmt_bs_forecaster.accounting_wmt import StructuralLayer
from wmt_bs_forecaster.types_wmt import DriversWMT, PoliciesWMT, PrevStateWMT

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

# --- Structural layer fixture -------------------------------------------------
@pytest.fixture
def struct():
    """Fresh WMT StructuralLayer per test."""
    return StructuralLayer()
