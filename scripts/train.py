"""
train.py

End-to-end training orchestration for the balance sheet forecaster.

This script does three things:
1. Loads REAL quarterly financial data for multiple tickers (YahooFinancialsLoader).
2. Splits data:
   - B split (some firms for training, some firms held out as external test)
   - T split (earlier quarters vs later quarters for forecasting evaluation)
3. Runs a smoke/unit test on DummyData to verify that the core model +
   training step + losses all work even if Yahoo is unavailable.

Keep this as a script for execution run, and do not import it into model code.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import tensorflow as tf

from balance_sheet_forecaster.data import YahooFinancialsLoader, DummyData
from balance_sheet_forecaster.model import AccountingModel
from balance_sheet_forecaster.types import Policies, PrevState
from balance_sheet_forecaster.losses import train_step, statement_fit_loss
from balance_sheet_forecaster.config import Config
from balance_sheet_forecaster.utils_logging import RunLogger, set_all_seeds


config = Config.from_cli()
set_all_seeds(config.seed)
logger = RunLogger(config.output_dir)

# Log config + tickers used
logger.write_meta(config, tickers_used, extras={"horizon_quarters": config.horizon_quarters})


# ----------------------------------------------------------------------
# Helpers for splitting data
# ----------------------------------------------------------------------

def subset_batch(
    idxs: List[int],
    features: tf.Tensor,
    policies: Policies,
    prev_state: PrevState,
    targets: Dict[str, tf.Tensor],
) -> Tuple[tf.Tensor, Policies, PrevState, Dict[str, tf.Tensor]]:
    """
    Select a subset of tickers along the batch axis B.

    idxs: list of batch indices to keep, e.g. [0,1,2] for training firms
    features: [B, T, F]
    policies: Policies with each field [B, T, 1]
    prev_state: PrevState with each field [B, 1]
    targets: dict[str -> tf.Tensor [B, T, 1]]

    returns sliced (features_sub, policies_sub, prev_sub, targets_sub)
    """
    feat_sub = tf.gather(features, idxs, axis=0)

    pol_sub = Policies(
        inflation=tf.gather(policies.inflation, idxs, axis=0),
        real_rate=tf.gather(policies.real_rate, idxs, axis=0),
        tax_rate=tf.gather(policies.tax_rate, idxs, axis=0),
        min_cash_ratio=tf.gather(policies.min_cash_ratio, idxs, axis=0),
        payout_ratio=tf.gather(policies.payout_ratio, idxs, axis=0),
    )

    prev_sub = PrevState(
        cash=tf.gather(prev_state.cash, idxs, axis=0),
        st_investments=tf.gather(prev_state.st_investments, idxs, axis=0),
        st_debt=tf.gather(prev_state.st_debt, idxs, axis=0),
        lt_debt=tf.gather(prev_state.lt_debt, idxs, axis=0),
        ar=tf.gather(prev_state.ar, idxs, axis=0),
        ap=tf.gather(prev_state.ap, idxs, axis=0),
        inventory=tf.gather(prev_state.inventory, idxs, axis=0),
        nfa=tf.gather(prev_state.nfa, idxs, axis=0),
        equity=tf.gather(prev_state.equity, idxs, axis=0),
    )

    targs_sub = {k: tf.gather(v, idxs, axis=0) for k, v in targets.items()}

    return feat_sub, pol_sub, prev_sub, targs_sub


def time_split(
    features: tf.Tensor,
    policies: Policies,
    prev_state: PrevState,
    targets: Dict[str, tf.Tensor],
    t_train_horizon: int,
):
    """
    Slice each ticker's time axis T into train window and validation (future) window.

    features: [B, T, F]
    policies: Policies with each [B, T, 1]
    prev_state: PrevState with each [B, 1]
    targets: dict[str -> [B, T, 1]]
    t_train_horizon: int, number of timesteps from the start to use for training

    returns:
        (feat_tr, pol_tr, prev_tr, targs_tr,
         feat_val, pol_val, prev_val, targs_val)
    """

    feat_tr = features[:, :t_train_horizon, :]
    feat_val = features[:, t_train_horizon:, :]

    pol_tr = Policies(
        inflation=policies.inflation[:, :t_train_horizon, :],
        real_rate=policies.real_rate[:, :t_train_horizon, :],
        tax_rate=policies.tax_rate[:, :t_train_horizon, :],
        min_cash_ratio=policies.min_cash_ratio[:, :t_train_horizon, :],
        payout_ratio=policies.payout_ratio[:, :t_train_horizon, :],
    )
    pol_val = Policies(
        inflation=policies.inflation[:, t_train_horizon:, :],
        real_rate=policies.real_rate[:, t_train_horizon:, :],
        tax_rate=policies.tax_rate[:, t_train_horizon:, :],
        min_cash_ratio=policies.min_cash_ratio[:, t_train_horizon:, :],
        payout_ratio=policies.payout_ratio[:, t_train_horizon:, :],
    )

    # Version 0: reuse same prev_state for both splits.
    # Later for true rollout forecasting, you'll propagate prev_state forward.
    prev_tr = prev_state
    prev_val = prev_state

    targs_tr = {k: v[:, :t_train_horizon, :] for k, v in targets.items()}
    targs_val = {k: v[:, t_train_horizon:, :] for k, v in targets.items()}

    return (
        feat_tr, pol_tr, prev_tr, targs_tr,
        feat_val, pol_val, prev_val, targs_val,
    )


# ----------------------------------------------------------------------
# Evaluation helper
# ----------------------------------------------------------------------

def evaluate_fit(
    model: AccountingModel,
    features: tf.Tensor,
    policies: Policies,
    prev_state: PrevState,
    targets: Dict[str, tf.Tensor],
) -> float:
    """
    Run the model forward (no grad), compute statement_fit_loss
    on the provided slice. Returns scalar float.
    """
    pred_statements, _pred_drivers = model(
        features, policies, prev_state, training=False
    )
    val_loss = statement_fit_loss(pred_statements, targets)
    return float(val_loss.numpy())


# ----------------------------------------------------------------------
# Smoke test on DummyData (unit-style check, not economic training)
# ----------------------------------------------------------------------

def smoke_test_dummy():
    """
    1. Create a tiny synthetic dataset (DummyData).
    2. Run a few train steps.
    3. Confirm shapes/loss flow without relying on Yahoo/network.

    This is what you'd run in CI.
    """
    print("=== dummy smoke test ===")
    tf.keras.utils.set_random_seed(123)

    dummy = DummyData(B=2, T=12, F=8)

    feats = dummy.features()        # [2,12,8]
    pol   = dummy.policies()        # Policies with [2,12,1] tensors
    prev  = dummy.prev()            # PrevState with [2,1] tensors
    targs = dummy.targets()         # dict[str->[2,12,1]]

    model = AccountingModel(hidden=64)
    opt = tf.keras.optimizers.Adam(1e-3)

    for step in range(10):
        loss, stm_pred, drv_pred = train_step(
            model, opt,
            feats, pol, prev,
            targs,
            w_acct=1e-5,
            w_smooth=1e-3,
            weights_map=None,
        )

        if step % 5 == 0:
            print(
                f"[dummy] step={step:02d} "
                f"loss={float(loss.numpy()):.4f} "
                f"assets≈liab+equity gap="
                f"{float(tf.reduce_mean(tf.abs(stm_pred.assets - stm_pred.liab_plus_equity)).numpy()):.6f}"
            )

    print("=== dummy smoke test done ===\n")


# ----------------------------------------------------------------------
# Real-data training and evaluation loop
# ----------------------------------------------------------------------

def train_on_real_data():
    """
    Pulls actual quarterly financials for multiple tickers,
    does B split and T split,
    trains on the training slice,
    and prints validation metrics on:
      (a) future quarters of the same firms (time holdout)
      (b) completely held-out firms (firm holdout)
    """

    print("=== real-data training start ===")

    # ------------------------------------------------------------------
    # 1. Load real fundamentals from Yahoo
    # ------------------------------------------------------------------
    # Pick tickers with decent history. You can add/remove here.
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    loader = YahooFinancialsLoader(
        tickers=TICKERS,
        horizon_quarters=8,  # require at least 8 most recent quarters
    )
    (
        features_all,      # [B,T,F]
        policies_all,      # Policies
        prev_all,          # PrevState
        targets_all,       # dict[str->[B,T,1]]
        tickers_used,      # list[str] actually kept after filtering
    ) = loader.all()

    print(f"tickers_used (B={len(tickers_used)}): {tickers_used}")
    B_total = len(tickers_used)

    if B_total < 3:
        raise RuntimeError(
            "Need at least 3 tickers for a reasonable train/held-out split."
        )

    # ------------------------------------------------------------------
    # 2. B split: choose which firms to train on vs hold out
    # ------------------------------------------------------------------
    # We'll just do a simple split: first 3 train, rest external test.
    train_firms_idx = list(range(min(3, B_total)))    # e.g. [0,1,2]
    external_idx    = list(range(min(3, B_total), B_total))  # e.g. [3,4,...]

    (
        features_trainB,
        policies_trainB,
        prev_trainB,
        targets_trainB,
    ) = subset_batch(
        train_firms_idx,
        features_all,
        policies_all,
        prev_all,
        targets_all,
    )

    if external_idx:
        (
            features_extB,
            policies_extB,
            prev_extB,
            targets_extB,
        ) = subset_batch(
            external_idx,
            features_all,
            policies_all,
            prev_all,
            targets_all,
        )
    else:
        features_extB = None
        policies_extB = None
        prev_extB = None
        targets_extB = None

    # ------------------------------------------------------------------
    # 3. T split for the training firms:
    #    Use first T_train quarters for training, hold out last quarters
    #    to simulate forecasting forward in time.
    # ------------------------------------------------------------------
    # features_trainB is [B_train, T, F]
    T_total = int(features_trainB.shape[1])
    # We'll keep last 2 quarters for validation and train on the rest:
    T_train = max(T_total - 2, 1)

    (
        feat_tr,
        pol_tr,
        prev_tr,
        targs_tr,
        feat_val,
        pol_val,
        prev_val,
        targs_val,
    ) = time_split(
        features_trainB,
        policies_trainB,
        prev_trainB,
        targets_trainB,
        T_train,
    )

    # ------------------------------------------------------------------
    # 4. Initialize model + optimizer
    # ------------------------------------------------------------------
    tf.keras.utils.set_random_seed(42)
    model = AccountingModel(hidden=64)
    opt = tf.keras.optimizers.Adam(1e-3)

    # ------------------------------------------------------------------
    # 5. Training loop on the TRAIN slice (feat_tr / pol_tr / targs_tr)
    # ------------------------------------------------------------------
    STEPS = 500
    for step in range(STEPS):
        # You can anneal the accounting guardrail weight over steps if you want.
        w_acct = 1e-5
        w_smooth = 1e-3

        loss, stm_pred, drv_pred = train_step(
            model,
            opt,
            feat_tr,
            pol_tr,
            prev_tr,
            targs_tr,
            w_acct=w_acct,
            w_smooth=w_smooth,
            weights_map=None,
        )

        # Occasionally compute validation metrics
        if step % 100 == 0 or step == STEPS - 1:
            # (a) time holdout performance on the SAME firms
            val_fit_internal = evaluate_fit(
                model,
                feat_val,
                pol_val,
                prev_val,
                targs_val,
            )

            # (b) generalization to completely held-out firms (B split)
            if features_extB is not None:
                val_fit_external = evaluate_fit(
                    model,
                    features_extB,
                    policies_extB,
                    prev_extB,
                    targets_extB,
                )
            else:
                val_fit_external = float("nan")

            print(
                f"[real] step={step:04d} "
                f"train_loss={float(loss.numpy()):.4f} "
                f"val_internal={val_fit_internal:.4f} "
                f"val_external={val_fit_external:.4f}"
            )

    print("=== real-data training done ===\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Run dummy smoke test (no network, shape/grad sanity)
    smoke_test_dummy()

    # 2. Train/eval on real Yahoo data with B and T splits
    train_on_real_data()
