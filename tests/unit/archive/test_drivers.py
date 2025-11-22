import numpy as np
import tensorflow as tf
import pytest

from balance_sheet_forecaster.drivers import DriverHead
from balance_sheet_forecaster.types import Drivers
from dataclasses import fields

# --- Global test tolerances ---------------------------------------------------
RTOL_DEFAULT = 1e-6
ATOL_DEFAULT = 1e-6

# ---- small helpers ---------------------------------------------------------

def _rand_features(B, T, F, seed=123):
    rng = tf.random.Generator.from_seed(seed)
    return rng.uniform(shape=[B, T, F], minval=0.0, maxval=1.0, dtype=tf.float32)

def _assert_bt1(name, x, B, T):
    assert isinstance(x, tf.Tensor), f"{name} not a tensor"
    assert x.shape == (B, T, 1), f"{name} shape {x.shape} != ({B},{T},1)"
    assert x.dtype == tf.float32, f"{name} dtype {x.dtype} != float32"

def assert_drivers_close(a, b, rtol=1e-6, atol=1e-6):
    assert type(a) is type(b), f"type mismatch: {type(a)} vs {type(b)}"
    for f in fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        # convert TF tensors to numpy
        if isinstance(va, tf.Tensor): va = va.numpy()
        if isinstance(vb, tf.Tensor): vb = vb.numpy()
        ok = np.allclose(va, vb, rtol=rtol, atol=atol)
        assert ok, f"field {f.name} differs beyond tol"

def assert_tensors_allclose(a, b, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT, msg=None):
    """np.allclose for TF/NumPy/scalars with a clean assertion message."""
    if isinstance(a, tf.Tensor): a = a.numpy()
    if isinstance(b, tf.Tensor): b = b.numpy()
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError(msg or f"values differ beyond rtol={rtol}, atol={atol}")

def assert_dataclass_tensors_close(a, b, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """
    Compare two dataclass instances field-by-field, allowing TF tensors or NumPy arrays.
    Raises AssertionError on the first mismatch (with the field name in the message).
    """
    if type(a) is not type(b):
        raise AssertionError(f"type mismatch: {type(a)} vs {type(b)}")
    for f in fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        try:
            assert_tensors_allclose(
                va, vb, rtol=rtol, atol=atol,
                msg=f"field {f.name} differs beyond tolerance",
            )
        except AssertionError as e:
            raise AssertionError(str(e))

# ---- tests -----------------------------------------------------------------

def test_forward_shapes_and_types():
    B, T, F = 3, 5, 8
    feats = _rand_features(B, T, F)
    model = DriverHead(hidden=32)

    out: Drivers = model(feats, training=False)

    _assert_bt1("price", out.price, B, T)
    _assert_bt1("volume", out.volume, B, T)
    _assert_bt1("dso_days", out.dso_days, B, T)
    _assert_bt1("dpo_days", out.dpo_days, B, T)
    _assert_bt1("dio_days", out.dio_days, B, T)
    _assert_bt1("capex", out.capex, B, T)
    _assert_bt1("stlt_split", out.stlt_split, B, T)

def test_domain_constraints_respected():
    B, T, F = 2, 4, 6
    feats = _rand_features(B, T, F)
    model = DriverHead(hidden=16)

    d: Drivers = model(feats, training=False)

    # strictly positive where required
    assert tf.reduce_min(d.price) > 0.0
    assert tf.reduce_min(d.volume) > 0.0
    assert tf.reduce_min(d.capex) >= 0.0

    # days bounds: >= 5 (softplus + 5.0 in the head)
    eps = 1e-6
    assert tf.reduce_min(d.dso_days) >= 5.0 - eps
    assert tf.reduce_min(d.dpo_days) >= 5.0 - eps
    assert tf.reduce_min(d.dio_days) >= 5.0 - eps

    # split in [0,1]
    assert tf.reduce_min(d.stlt_split) >= 0.0 - eps
    assert tf.reduce_max(d.stlt_split) <= 1.0 + eps

def test_reproducibility_with_seed():
    """
    With a fixed global seed, two independently-created models
    produce identical outputs on the same input.
    """
    B, T, F = 2, 3, 7
    feats = _rand_features(B, T, F, seed=777)

    # Build model A under seed S
    tf.keras.utils.set_random_seed(42)
    model_a = DriverHead(hidden=32)
    out_a = model_a(feats, training=False)

    # Build model B under the same seed S -> same initial weights
    tf.keras.utils.set_random_seed(42)
    model_b = DriverHead(hidden=32)
    out_b = model_b(feats, training=False)

    # Compare all fields tensor-wise with tolerances
    assert_dataclass_tensors_close(out_a, out_b, rtol=1e-6, atol=1e-6)

def test_gradients_flow():
    """
    Ensure the head is trainable end-to-end: d(sum(outputs))/d(theta) exists.
    """
    B, T, F = 2, 4, 5
    feats = _rand_features(B, T, F, seed=999)
    model = DriverHead(hidden=16)

    with tf.GradientTape() as tape:
        out = model(feats, training=True)
        # sum a few outputs to make a scalar loss
        loss = (
            tf.reduce_sum(out.price) +
            tf.reduce_sum(out.volume) +
            tf.reduce_sum(out.capex) +
            tf.reduce_sum(out.stlt_split)
        )
    grads = tape.gradient(loss, model.trainable_variables)

    assert all(g is not None for g in grads), "Some gradients are None"
    # also check not-all-zero to catch dead activations
    nonzero = [tf.reduce_any(tf.not_equal(g, 0.0)).numpy() for g in grads]
    assert any(nonzero), "All gradients are zero"
