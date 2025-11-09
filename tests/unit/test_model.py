import numpy as np
import tensorflow as tf
import pytest

from balance_sheet_forecaster.model import BalanceSheetForecastModel
from balance_sheet_forecaster.types import Drivers, Statements

def _assert_dataclass_bt1_shapes(dc_obj, B, T, name=""):
    # Every field in Drivers/Statements must be [B, T, 1]
    for field_name, value in dc_obj.__dict__.items():
        assert isinstance(value, tf.Tensor), f"{name}.{field_name} is not a tf.Tensor"
        assert value.shape == (B, T, 1), f"{name}.{field_name} shape {value.shape} != {(B, T, 1)}"

def test_forward_shapes_and_identity(mk_policies, mk_prev, set_seeds):
    B, T, F = 2, 4, 8
    features = tf.random.uniform([B, T, F], 0.0, 1.0, dtype=tf.float32)
    policies = mk_policies(B, T)  # default policy tensors [B,T,1]
    prev = mk_prev(B)             # default prev state [B,1]

    model = BalanceSheetForecastModel(hidden=32)
    stm, drv = model(features, policies=policies, prev=prev, training=False)

    # Shapes
    assert isinstance(stm, Statements)
    assert isinstance(drv, Drivers)
    _assert_dataclass_bt1_shapes(stm, B, T, "stm")
    _assert_dataclass_bt1_shapes(drv, B, T, "drivers")

    # Soft accounting identity check (the StructuralLayer enforces it internally,
    # but we keep a small external tolerance here as a sanity guard.)
    diff = tf.abs(stm.assets - stm.liab_plus_equity)
    assert float(tf.reduce_max(diff).numpy()) < 1e-4

def test_train_vs_eval_consistency(mk_policies, mk_prev, set_seeds):
    # No Dropout/BatchNorm in the stack, so training flag should not change outputs.
    B, T, F = 2, 3, 7
    features = tf.random.uniform([B, T, F], 0.0, 1.0, dtype=tf.float32)
    policies = mk_policies(B, T)
    prev = mk_prev(B)

    model = BalanceSheetForecastModel(hidden=16)

    stm_eval, drv_eval = model(features, policies=policies, prev=prev, training=False)
    stm_train, drv_train = model(features, policies=policies, prev=prev, training=True)

    for k in stm_eval.__dict__.keys():
        a = stm_eval.__dict__[k].numpy()
        b = stm_train.__dict__[k].numpy()
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6), f"Statements field {k} differs between eval/train."

    for k in drv_eval.__dict__.keys():
        a = drv_eval.__dict__[k].numpy()
        b = drv_train.__dict__[k].numpy()
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6), f"Drivers field {k} differs between eval/train."

def test_gradients_flow(mk_policies, mk_prev, set_seeds):
    # Ensure end-to-end differentiability: a simple scalar loss should backprop.
    B, T, F = 2, 5, 6
    features = tf.random.uniform([B, T, F], 0.0, 1.0, dtype=tf.float32)
    policies = mk_policies(B, T)
    prev = mk_prev(B)

    model = BalanceSheetForecastModel(hidden=12)

    with tf.GradientTape() as tape:
        stm, drv = model(features, policies=policies, prev=prev, training=True)
        # Pick a simple scalar loss that depends on model outputs.
        # sales depends on drivers (price*volume), so gradients should flow through the stack.
        loss = tf.reduce_mean(stm.sales)

    grads = tape.gradient(loss, model.trainable_variables)
    # No gradient should be None; at least some should be non-zero.
    assert all(g is not None for g in grads), "Found None gradients; graph may be broken."
    nonzero = [tf.reduce_sum(tf.abs(g)).numpy() for g in grads]
    assert any(v > 0 for v in nonzero), "All gradients are zero; model might be disconnected."

def test_reproducibility_with_seed(mk_policies, mk_prev):
    # With a fixed TF seed, two fresh models should produce identical outputs on same input.
    B, T, F = 2, 3, 5
    features = tf.random.uniform([B, T, F], 0.0, 1.0, dtype=tf.float32)
    policies = mk_policies(B, T)
    prev = mk_prev(B)

    tf.keras.utils.set_random_seed(42)
    m1 = BalanceSheetForecastModel(hidden=10)
    s1, d1 = m1(features, policies=policies, prev=prev, training=False)

    tf.keras.utils.set_random_seed(42)
    m2 = BalanceSheetForecastModel(hidden=10)
    s2, d2 = m2(features, policies=policies, prev=prev, training=False)

    for k in s1.__dict__.keys():
        assert np.allclose(s1.__dict__[k].numpy(), s2.__dict__[k].numpy(), rtol=1e-6, atol=1e-6), f"Statements {k} mismatch."
    for k in d1.__dict__.keys():
        assert np.allclose(d1.__dict__[k].numpy(), d2.__dict__[k].numpy(), rtol=1e-6, atol=1e-6), f"Drivers {k} mismatch."
