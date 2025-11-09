from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import tensorflow as tf

from balance_sheet_forecaster.types import Statements, Drivers, Policies, PrevState
from balance_sheet_forecaster.model import BalanceSheetForecastModel


def advance_prev(stm_last: Statements) -> PrevState:
    """
    Advance the PrevState by one period using the last Statements.

    Args:
        stm_last: Statements
            The Statements at the last time step, each field shaped [B, 1].

    Returns:
        PrevState
            The advanced PrevState for the next period.
    """
    return PrevState(
        cash=stm_last.cash[:, -1, :],
        st_investments=stm_last.st_investments[:, -1, :],
        st_debt=stm_last.st_debt[:, -1, :],
        lt_debt=stm_last.lt_debt[:, -1, :],
        ar=stm_last.ar[:, -1, :],
        ap=stm_last.ap[:, -1, :],
        inventory=stm_last.inventory[:, -1, :],
        nfa=stm_last.nfa[:, -1, :],
        equity=stm_last.equity[:, -1, :],
    )


def _slice_policies(policies: Policies, t: int) -> Policies:
    """
    Slice the Policies dataclass to get the policies at time step t.

    Args:
        policies: Policies
            The full Policies dataclass with fields.
        t: int
            The time step to slice.

    Returns:
        Policies
            The Policies dataclass sliced to [:, t, :].
    """
    return Policies(
        inflation=policies.inflation[:, t, :],
        real_rate=policies.real_rate[:, t, :],
        tax_rate=policies.tax_rate[:, t, :],
        min_cash_ratio=policies.min_cash_ratio[:, t, :],
        payout_ratio=policies.payout_ratio[:, t, :],
    )


def rollout(
        model: BalanceSheetForecastModel,
        features_roll: tf.Tensor,
        policies_roll: Policies,
        prev_last: PrevState,
        steps: int,
        return_drivers: bool = False,
) -> Tuple[Statements, Optional[Drivers]]:
    """
    Walk-forward forecast for `steps` periods.

    At step t we feed the model the slice [:, :t+1, :] so the GRU sees all history up to t,
    then advance PrevState using the last predicted balances.

    Returns:
      Statements over the forecast horizon [B, steps, 1] (and Drivers if requested).
    """
    
    stm_acc: Dict[str, List[tf.Tensor]] = {name: [] for name in Statements.__dataclass_fields__.keys()}
    drv_acc: Dict[str, List[tf.Tensor]] = {name: [] for name in Drivers.__dataclass_fields__.keys()}

    prev = prev_last

    # Defensive clamp: steps must not exceed the available T in features/policies
    T_future = int(features_roll.shape[1])
    if steps > T_future:
        raise ValueError(f"`steps` ({steps}) exceeds available features length ({T_future}).")

    for t in range(steps):
        # history slice up to and including t (shape [:, t+1, ...])
        feats_t = features_roll[:, :t + 1, :]
        pol_t = _slice_policies(policies_roll, t + 1)

        # forward pass (no training)
        stm_t, drv_t = model(feats_t, pol_t, prev, training=False)

        # take last step predictions and collect
        last = {k: getattr(stm_t, k)[:, -1:, :] for k in stm_acc.keys()}
        for k, v in last.items():
            stm_acc[k].append(v)

        if return_drivers:
            last_drv = {k: getattr(drv_t, k)[:, -1:, :] for k in drv_acc.keys()}
            for k, v in last_drv.items():
                drv_acc[k].append(v)

        # advance PrevState to the end-of-period balances we just predicted
        prev = advance_prev(stm_t)

    # Stack lists into tensors along time axis -> [B, steps, 1]
    stm_stacked = {k: tf.concat(v_list, axis=1) for k, v_list in stm_acc.items()}
    stm_out = Statements(**stm_stacked)

    if return_drivers:
        drv_stacked = {k: tf.concat(v_list, axis=1) for k, v_list in drv_acc.items()}
        return stm_out, Drivers(**drv_stacked)

    return stm_out, None