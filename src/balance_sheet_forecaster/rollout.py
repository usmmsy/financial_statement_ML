from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Tuple, Dict, Optional, List

import tensorflow as tf

from balance_sheet_forecaster.types import Statements, Drivers, Policies, PrevState
from balance_sheet_forecaster.model import BalanceSheetForecastModel


def advance_prev(stm: Statements, t: Optional[int] = None) -> PrevState:
    """
    Build a PrevState from a Statements object.

    If `stm` is per-step ([B,1,1] per field), `t` can be omitted.
    If `stm` spans multiple steps ([B,T,1] per field), pass the 0-based index `t`.
    """
    def pick(x: tf.Tensor) -> tf.Tensor:
        # x is [B, T, 1] or [B, 1, 1]
        if t is None:
            return x[:, -1:, :]          # last step
        else:
            return x[:, t:t+1, :]        # specific step
    return PrevState(
        cash=pick(stm.cash),
        st_investments=pick(stm.st_investments),
        st_debt=pick(stm.st_debt),
        lt_debt=pick(stm.lt_debt),
        ar=pick(stm.ar),
        ap=pick(stm.ap),
        inventory=pick(stm.inventory),
        nfa=pick(stm.nfa),
        equity=pick(stm.equity),
    )


# Policies dataclass slicer, no longer needed because Policies dataclass is now subscriptable
# def _slice_policies(policies: Policies, t: int) -> Policies:
#     """Return a Policies object with every available field sliced to step t: [:, t:t+1, :]."""
#     d = asdict(policies)  # {"inflation": Tensor or None, ...}
#     def sel(v):
#         return None if v is None else v[:, t:t+1, :]
#     d = {k: sel(v) for k, v in d.items()}
#     return Policies(**d)


def rollout(
    model: BalanceSheetForecastModel,
    features: tf.Tensor,                  # [B, T, F]
    policies_roll: Policies,              # each field [B, T, 1]
    prev_last: PrevState,                 # each field [B, 1]
    steps: Optional[int] = None,
    training: bool = False,
    return_drivers: bool = False,
) -> Tuple[Statements, Optional[List[Drivers]]]:
    """
    Roll forward one step at a time and stack Statements along time.

    Returns:
        stm_seq: Statements with each field shaped [B, steps, 1]
        drv_seq: list of per-step Drivers (length=steps) or None if not requested
    """
    # Resolve horizon
    B, T, F = features.shape
    steps = steps or T

    prev = prev_last
    stms: List[Statements] = []
    drvs: List[Drivers] = []

    for t in range(steps):
        feats_t = features[:, t:t+1, :]          # [B, 1, F]
        pol_t = policies_roll[:, t:t+1, :]  # fields [B, 1, 1] (or None)

        # NOTE: non-tensor arguments (Policies/PrevState) passed as keywords
        stm_t, drv_t = model(feats_t, policies=pol_t, prev=prev, training=training)
        stms.append(stm_t)
        if return_drivers:
            drvs.append(drv_t)

        # stm_t is already [B,1,1] per field; advance to next PrevState
        prev = advance_prev(stm_t)

    # Stack Statements over time along T
    def cat(name: str) -> tf.Tensor:
        return tf.concat([getattr(s, name) for s in stms], axis=1)  # [B, steps, 1]

    stm_seq = Statements(
        sales=cat("sales"),
        cogs=cat("cogs"),
        opex=cat("opex"),
        ebit=cat("ebit"),
        interest=cat("interest"),
        tax=cat("tax"),
        net_income=cat("net_income"),
        cash=cat("cash"),
        ar=cat("ar"),
        ap=cat("ap"),
        inventory=cat("inventory"),
        st_investments=cat("st_investments"),
        st_debt=cat("st_debt"),
        lt_debt=cat("lt_debt"),
        nfa=cat("nfa"),
        equity=cat("equity"),
        ncb=cat("ncb"),
    )

    drv_seq: Optional[List[Drivers]] = drvs if return_drivers else None

    return stm_seq, drv_seq
