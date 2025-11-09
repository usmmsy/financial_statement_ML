from __future__ import annotations
from typing import Tuple
import tensorflow as tf

from balance_sheet_forecaster.types import (
    Drivers, 
    Policies, 
    PrevState, 
    Statements,
)
from balance_sheet_forecaster.drivers import DriverHead
from balance_sheet_forecaster.accounting import StructuralLayer


# Combined Model Wrapper
class BalanceSheetForecastModel(tf.keras.Model):
    """
    BalanceSheetForecastModel
    High-level wrapper that combines:
        1. DriverHead: RNN to learn behavioral dynamics / managerial drivers
        2. StructuralLayer: deterministic accounting / balance sheet roll-forward engine
    
    Call pattern:
        stm, drivers = model(
            features,    # [B, T, F]
            policies,    # Policies dataclass
            prev,        # PrevState dataclass
            training=bool
        )

    Args to call:
        features: tf.Tensor of shape [B, T, F]
            Model inputs per period, e.g. macro signals, lagged KPIs, 
            sales indicators, etc. This is what the DriverHead uses
            to predict behavioral drivers.

        policies: Policies
            Exogenous management policies / macro assumptions per period,
            e.g. tax rates, target cash ratios, inflation, etc.
            These are NOT learned.

        prev: PrevState
            The closing balance sheet from the last known period,
            shaped [B, 1] for each field (cash, debt, equity, etc).
            This anchors the forecast and is used by StructuralLayer 
            to compute interest on prior balances and to roll forward.

        training: bool
            Standard Keras training flag.

    Returns:
        stm: Statements
            Model-generated financial statements (IS + cash budget + BS)
            per period in the forecast horizon. Each field is [B, T, 1].

        drivers: Drivers
            Behavioral drivers predicted by DriverHead, per period.
            Each field is [B, T, 1].

    Important notes:
        This model does NOT itself iterate PrevState forward across timesteps.
        For multi-period sequential simulation (forecasting into the future
        where each period's ending balances become the next period's prev),
        use the rollout utilities in rollout.py.
    """

    def __init__(self, hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.driver_head = DriverHead(hidden)
        self.struct = StructuralLayer()

    def call(self, 
             features: tf.Tensor, 
             policies: Policies,
             prev: PrevState,
             training: bool = False,
        ) -> Tuple[Statements, Drivers]:

        # Optional safety: remove comment for rank checks
        # tf.debugging.assert_rank(features, 3, message="features must be [B, T, F]")
        # tf.debugging.assert_rank(policies.inflation, 3, message="policies fields must be [B, T, 1]")
        # tf.debugging.assert_rank(prev.cash, 2, message="prev fields must be [B, 1]")
        
        # 1. Predict behavioral / operating / financing drivers over the horizon
        drivers = self.driver_head(features, training=training)

        # 2. Apply deterministic accounting logic using 
        #    predicted drivers + policies + prior balance sheet
        stm = self.struct(drivers=drivers, policies=policies, prev=prev, training=training)
        
        return stm, drivers

