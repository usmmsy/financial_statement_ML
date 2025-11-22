import tensorflow as tf
from .types_wmt import StatementsWMT


def masked_l1(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """Mean absolute error ignoring NaNs in target."""
    mask = tf.logical_not(tf.math.is_nan(target))
    diff = tf.abs(pred - tf.where(mask, target, pred))  # zero where nan
    # keep only valid entries
    valid = tf.boolean_mask(diff, mask)
    return tf.reduce_mean(valid) if tf.size(valid) > 0 else tf.constant(0.0, dtype=pred.dtype)


def identity_gap_loss(statements: StatementsWMT) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(statements.assets - statements.liab_plus_equity))


def retained_earnings_consistency_loss(statements: StatementsWMT) -> tf.Tensor:
    # retained_end - retained_{t-1} should equal net_income_t - dividends_t
    retained = statements.retained_earnings  # [B,T,1]
    ni = statements.net_income
    div = statements.dividends
    # Differences across time
    delta_retained = retained[:, 1:, :] - retained[:, :-1, :]
    expected = ni[:, 1:, :] - div[:, 1:, :]
    return tf.reduce_mean(tf.abs(delta_retained - expected))


def wmt_fit_loss(statements: StatementsWMT,
                 targets: dict,
                 weights: dict | None = None,
                 include_identity: bool = True,
                 include_retained: bool = True,
                 identity_weight: float = 1.0,
                 retained_weight: float = 1.0) -> dict:
    """
    Aggregate Walmart-specific losses.

    targets: mapping from line name to target tensor [B,T,1]
    weights: optional per-line scalar weights
    Returns dict with individual components and 'total'.
    """
    weights = weights or {}
    comps = []
    out = {}

    def add(name: str, loss_tensor: tf.Tensor):
        w = float(weights.get(name, 1.0))
        val = loss_tensor * w
        out[name] = val
        comps.append(val)

    # Core fit lines (only if present in targets)
    for key, tensor in targets.items():
        if not hasattr(statements, key):
            continue
        pred = getattr(statements, key)
        add(f"fit_{key}", masked_l1(pred, tf.cast(tensor, pred.dtype)))

    if include_identity:
        add("identity_gap", identity_gap_loss(statements) * identity_weight)
    if include_retained:
        add("retained_consistency", retained_earnings_consistency_loss(statements) * retained_weight)

    total = tf.add_n(comps) if comps else tf.constant(0.0, dtype=tf.float32)
    out["total"] = total
    return out
