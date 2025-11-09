import numpy as np
import tensorflow as tf
import pytest

from balance_sheet_forecaster.model import BalanceSheetForecastModel
from balance_sheet_forecaster.training import train_step, eval_fit
from balance_sheet_forecaster.data import DummyData
from balance_sheet_forecaster.losses import statement_fit_loss

def _with_eager(fn, *args, **kwargs):
    prev = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)
    try:
        return fn(*args, **kwargs)
    finally:
        tf.config.run_functions_eagerly(prev)

def test_train_step_reduces_loss(set_seeds):
    """
    Sanity: a few train steps should reduce the fit loss on synthetic data.
    """
    B, T, F = 2, 8, 10
    dd = DummyData(B=B, T=T, F=F)

    features = dd.features()
    policies = dd.policies()
    prev     = dd.prev()
    targets  = dd.targets()

    model = BalanceSheetForecastModel(hidden=16)
    if hasattr(model, "struct") and hasattr(model.struct, "hard_identity_check"):
        model.struct.hard_identity_check = False

    opt = tf.keras.optimizers.Adam(1e-2)

    # initial eval loss
    loss0, _, _ = _with_eager(
        eval_fit, model, features, policies=policies, prev=prev, targets=targets, weights_map=None
    )
    loss0 = float(loss0.numpy())

    # a few steps of training
    for _ in range(10):
        loss, _, _ = _with_eager(
            train_step, model, opt, features, policies=policies, prev=prev, targets=targets,
            w_acct=1e-5, w_smooth=1e-3, weights_map=None
        )

    # post-train eval loss
    loss1, _, _ = _with_eager(
        eval_fit, model, features, policies=policies, prev=prev, targets=targets, weights_map=None
    )
    loss1 = float(loss1.numpy())

    assert loss1 <= loss0 * 0.99  # at least 1% better (slack for randomness)


def test_eval_fit_matches_manual(set_seeds):
    """
    eval_fit() should match a manual forward + statement_fit_loss computation.
    """
    B, T, F = 1, 5, 6
    dd = DummyData(B=B, T=T, F=F)

    features = dd.features()
    policies = dd.policies()
    prev     = dd.prev()
    targets  = dd.targets()

    model = BalanceSheetForecastModel(hidden=8)
    if hasattr(model, "struct") and hasattr(model.struct, "hard_identity_check"):
        model.struct.hard_identity_check = False
        
    # eval_fit
    fit_eval, stm_eval, _ = _with_eager(
        eval_fit, model, features, policies=policies, prev=prev, targets=targets, weights_map=None
    )
    # manual
    stm_manual, _ = model(features, policies=policies, prev=prev, training=False)
    fit_manual = statement_fit_loss(stm_manual, targets, weights=None)

    assert np.allclose(float(fit_eval.numpy()), float(fit_manual.numpy()), rtol=1e-6, atol=1e-6)


def test_train_step_returns_reasonable_objects(set_seeds):
    """
    Contract: train_step returns (loss scalar, Statements, Drivers) with expected shapes.
    """
    B, T, F = 2, 6, 7
    dd = DummyData(B=B, T=T, F=F)

    features = dd.features()
    policies = dd.policies()
    prev     = dd.prev()
    targets  = dd.targets()

    model = BalanceSheetForecastModel(hidden=12)
    if hasattr(model, "struct") and hasattr(model.struct, "hard_identity_check"):
        model.struct.hard_identity_check = True
        
    opt = tf.keras.optimizers.Adam(1e-3)

    loss, stm, drv = _with_eager(
        train_step, model, opt, features, policies=policies, prev=prev, targets=targets,
        w_acct=1e-5, w_smooth=1e-3, weights_map=None
    )

    # loss is scalar
    assert np.isscalar(float(loss.numpy()))

    # key statement fields have [B,T,1]
    for t in [stm.sales, stm.cogs, stm.cash, stm.nfa, stm.equity]:
        assert t.shape == (B, T, 1)

    # key driver fields have [B,T,1]
    for d in [drv.price, drv.volume, drv.dso_days, drv.capex, drv.stlt_split]:
        assert d.shape == (B, T, 1)
