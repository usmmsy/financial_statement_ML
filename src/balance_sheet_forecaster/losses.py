from typing import Dict
from balance_sheet_forecaster.types import Statements, Drivers

import tensorflow as tf


# Losses + Regularizers

def mae(x: tf.Tensor) -> tf.Tensor:
    """
    Mean Absolute Error (MAE) loss.

    Input:
        x: [B, T, ...] or [B, T-1, ...]
    Output:
        scalar tensor
    """

    return tf.reduce_mean(tf.abs(x))

def statement_fit_loss(
        pred: Statements, 
        true: Dict[str, tf.Tensor], 
        weights: Dict[str, float] = None,
    ) -> tf.Tensor:
    """
    Fit loss on financial statement lines we care about.

    We compare predicted time series (from StructuralLayer) to provided
    historical/observed time series, line by line, using an L1 objective.

    Args:
        pred:
            Statements produced by the model.
            Each field is [B, T, 1].
        true:
            Dict[str, tf.Tensor] mapping line names -> ground truth tensors,
            each of shape [B, T, 1].
            You don't have to provide every line; missing keys are skipped.
        weights:
            Optional dict mapping line name -> scalar float weight.
            For example, you may want to weight balance sheet lines higher
            than income statement lines, because the model is balance-sheet first.

    Returns:
        Scalar loss (tf.Tensor).
    """

    weights = weights or {}
    loss = tf.constant(0.0, dtype=tf.float32)

    candidate_keys = [
        "sales", 
        "cogs", 
        "opex", 
        "ebit", 
        "net_income",
        "cash", 
        "ar", 
        "ap", 
        "inventory", 
        "st_investments",
        "st_debt", 
        "lt_debt", 
        "nfa", 
        "equity"
    ]

    for key in candidate_keys:
        if key in true:
            w = weights.get(key, 1.0)
            pred_line = getattr(pred, key)
            loss += w * mae(pred_line - true[key])

    return loss

def identity_guardrail(pred: Statements) -> tf.Tensor:
    """
    Soft penalty for violating Assets = Liabilities + Equity.

    This should be applied in training with a TINY coefficient
    (like 1e-5 down to 1e-7). The StructuralLayer is already designed
    to satisfy the identity, so this loss is mostly a safety rail and
    early stabilizer, not a crutch to 'force' balance.
    
    Args:
        pred: Statements object with model predictions.
    Returns:
        Identity deviation loss as a scalar tensor.
    """

    gap = tf.reduce_mean(tf.abs(pred.assets - pred.liab_plus_equity))
    # Keep tiny
    
    return gap

def smoothness_penalty(drivers: Drivers, lam: float = 1.0) -> tf.Tensor:
    """
    Penalize large temporal changes in driver variables to encourage smoothness.

    Args:
        drivers: Drivers object with model-generated driver time series.
        lam: Weighting factor for the smoothness penalty.
    Returns:
        Smoothness penalty as a scalar tensor.
    """

    def diff(x):
        return x[:, 1:, :] - x[:, :-1, :]
    
    total = 0.0
    for series in [drivers.price, drivers.volume, drivers.dso_days,
                  drivers.dpo_days, drivers.dio_days,
                  drivers.capex, drivers.stlt_split]:
        total += mae(diff(series))
    return lam * total
