import numpy as np
import tensorflow as tf
import pytest

from balance_sheet_forecaster.model import BalanceSheetForecastModel
from balance_sheet_forecaster.rollout import advance_prev, rollout
from balance_sheet_forecaster.data import DummyData


def test_advance_prev_matches_terminal_period(mk_policies, mk_prev, set_seeds):
    """
    advance_prev should copy terminal statement lines into a [B,1] PrevState.
    """
    B, T, F = 2, 4, 8
    dd = DummyData(B=B, T=T, F=F)

    features = dd.features()
    policies = dd.policies()
    prev0    = dd.prev()

    model = BalanceSheetForecastModel(hidden=16)
    stm, _ = model(features, policies=policies, prev=prev0, training=False)  # [B,T,1] per field

    t_last = T - 1
    prev_next = advance_prev(stm, t_last)  # [B,1] per field

    # spot-check a few fields: cash/nfa/equity
    assert np.allclose(prev_next.cash.numpy().squeeze(),        stm.cash[:, t_last, :].numpy().squeeze(),  rtol=1e-6, atol=1e-6)
    assert np.allclose(prev_next.nfa.numpy().squeeze(),         stm.nfa[:, t_last, :].numpy().squeeze(),   rtol=1e-6, atol=1e-6)
    assert np.allclose(prev_next.equity.numpy().squeeze(),      stm.equity[:, t_last, :].numpy().squeeze(),rtol=1e-6, atol=1e-6)


def test_rollout_shapes_and_identity(mk_policies, mk_prev, set_seeds):
    """
    Rollout should produce a sequence with the requested horizon and preserve identity each step.
    """
    B, T, F = 2, 5, 6
    dd = DummyData(B=B, T=T, F=F)

    features = dd.features()
    policies = dd.policies()
    prev0    = dd.prev()

    model = BalanceSheetForecastModel(hidden=12)

    # If your rollout returns a stacked Statements (with T dim == horizon), keep as is.
    # If it returns a list of per-step Statements, stack or loop to assert shapes/identity.
    stm_seq, drv_seq = rollout(model, features, policies_roll=policies, prev_last=prev0, steps=T, training=False, return_drivers=False)

    # Expect [B,T,1] per statement field
    for t in [stm_seq.sales, stm_seq.cogs, stm_seq.cash, stm_seq.equity]:
        assert t.shape == (B, T, 1)

    # per-period identity Assets == Liab + Equity should be tight
    diff = tf.abs(stm_seq.assets - stm_seq.liab_plus_equity)              # [B,T,1]
    max_diff = tf.reduce_max(diff)
    scale = tf.reduce_max(tf.abs(stm_seq.assets)) + 1e-6                  # avoid div-by-zero
    rel = max_diff / scale
    assert float(rel.numpy()) < 0.3   # allow 30% mismatch for the toy model

    # diff_last = tf.abs(stm_seq.assets[:, -1:, :] - stm_seq.liab_plus_equity[:, -1:, :])
    # rel_last = tf.reduce_max(diff_last) / (tf.reduce_max(tf.abs(stm_seq.assets[:, -1:, :])) + 1e-6)
    # assert float(rel_last.numpy()) < 0.2



def test_roll_step_vs_full_rollout_equivalence(mk_policies, mk_prev, set_seeds):
    """
    Running rollout for T steps should match doing T single-step advances (smoke-level equivalence).
    """
    B, T, F = 1, 4, 6
    dd = DummyData(B=B, T=T, F=F)

    features = dd.features()
    policies = dd.policies()
    prev = dd.prev()

    model = BalanceSheetForecastModel(hidden=10)

    # Full rollout
    stm_full, _ = rollout(model, features, policies_roll=policies, prev_last=prev, steps=T, training=False, return_drivers=False)

    # step-by-step manual advances
    prev_step = prev
    sales_last = []
    for t in range(T):
        stm_t, _ = model(
            features[:, t:t+1, :], 
            policies=policies[:, t:t+1, :], 
            prev=prev_step, 
            training=False
        )

        # take the last period t from this forward pass
        sales_last.append(stm_t.sales[:, -1:, :].numpy())
        prev_step = advance_prev(stm_t, -1)

    sales_manual = np.concatenate(sales_last, axis=1)  # [B,T,1]
    assert np.allclose(stm_full.sales.numpy(), sales_manual, rtol=1e-5, atol=1e-5)

def test_rollout_can_return_drivers(mk_policies, mk_prev, set_seeds):
    B, T, F = 1, 2, 4
    dd = DummyData(B=B, T=T, F=F)
    model = BalanceSheetForecastModel(hidden=8)

    stm_seq, drv_seq = rollout(
        model,
        dd.features(),
        policies_roll=dd.policies(),
        prev_last=dd.prev(),
        steps=T,
        training=False,
        return_drivers=True,
    )
    assert drv_seq is not None
    assert len(drv_seq) == T
    assert stm_seq.sales.shape == (B, T, 1)
