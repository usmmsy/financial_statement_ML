from __future__ import annotations
from typing import Dict, Optional, Tuple

import tensorflow as tf

from balance_sheet_forecaster.model import BalanceSheetForecastModel
from balance_sheet_forecaster.types import Statements, Drivers, Policies, PrevState
from balance_sheet_forecaster.losses import statement_fit_loss, identity_guardrail, smoothness_penalty


@tf.function
def train_step(
    model: BalanceSheetForecastModel,
    optimizer: tf.keras.optimizers.Optimizer,
    features: tf.Tensor,
    policies: Policies,
    prev: PrevState,
    targets: Dict[str, tf.Tensor],  # ground truth target statements
    *,  # keyword-only
    w_acct: float = 1e-5,   # weight for accounting identity guardrail
    w_smooth: float = 1e-3,  # weight for smoothness penalty
    weights_map: Optional[Dict[str, float]] = None, # line weights for fit loss
) -> Tuple[tf.Tensor, Statements, Drivers]:
    """
    Single optimization step: forward -> loss -> backprop.
    """

    with tf.GradientTape() as tape:
        pred_stm, pred_drv = model(
            features, policies=policies, prev=prev, training=True
        )
        fit = statement_fit_loss(pred_stm, targets, weights=weights_map)
        guard = identity_guardrail(pred_stm)
        smooth = smoothness_penalty(pred_drv)
        loss = fit + w_acct * guard + w_smooth * smooth

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, pred_stm, pred_drv


@tf.function
def eval_fit(
    model: BalanceSheetForecastModel,
    features: tf.Tensor,
    policies: Policies,
    prev: PrevState,
    targets: Dict[str, tf.Tensor],
    *,  # keyword-only
    weights_map: Optional[Dict[str, float]] = None,
) -> Tuple[tf.Tensor, Statements, Drivers]:
    """
    No-grad evaluation of fit loss.
    """

    pred_stm, pred_drv = model(
        features, policies=policies, prev=prev, training=False
    )
    fit = statement_fit_loss(pred_stm, targets, weights=weights_map)
    return fit, pred_stm, pred_drv

