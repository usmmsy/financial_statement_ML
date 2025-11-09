from __future__ import annotations
from balance_sheet_forecaster.types import Drivers
from typing import Optional

import tensorflow as tf


# Driver Head (trainable behavioral dynamics)


class DriverHead(tf.keras.Model):
    """
    DriverHead
    RNN-based model to predict behavioral drivers for each forecast period.
    
    Inputs:
        features: [B, T, F] 
            Tensor of input features per period
            1. lagged financials
            2. macro variables
            3. internal KPIs / operational signals
            4. management guidance features, etc.

        policy_hints: [B, T, H] (optional)
            Future constraints / guidance the business already knows
            (e.g. planned capex, target cash ratios, debt maturities, etc.)
            Currently unused, but could be concatenated to features later.

    Outputs:
        Drivers dataclass with predicted driver time series.
            price: selling price level / ASP
            volume: sales volume
            dso_days: days sales outstanding
            dpo_days: days payable outstanding
            dio_days: days inventory outstanding
            capex: capital expenditures
            stlt_split: fraction of new debt that is LT debt
    """

    def __init__(self, hidden=64, **kwargs):
        super().__init__(**kwargs)

        # Recurrent core: learns temporal behavior / policy dynamics
        self.rnn = tf.keras.layers.GRU(
            hidden, 
            return_sequences=True,
            name="driver_gru",
        )

        # Projection head: maps GRU hidden states to raw driver logits
        # NOTE: final Dense has no activation. We enforce domains manually below.
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden, activation='gelu', name="mlp_hidden"),
                tf.keras.layers.Dense(7, name="mlp_output") # 7 drivers
            ],
            name="driver_mlp",
        )

    def call(
            self, 
            features: tf.Tensor, 
            policy_hints: Optional[tf.Tensor] = None, 
            training: bool = False,
        ) -> Drivers:
        # features: [B, T, F]
        # policy_hints: [B, T, H] or None (currently unused)
        # output Drivers fields: [B, T, 1] each

        h = self.rnn(features, training=training)   # [B, T, hidden]
        raw = self.mlp(h, training=training)    # [B, T, 7]

        # Split out raw channels
        raw_price = raw[..., 0:1]
        raw_volume = raw[..., 1:2]
        raw_dso = raw[..., 2:3] # min 5 days
        raw_dpo = raw[..., 3:4] # min 5 days
        raw_dio = raw[..., 4:5] # min 5 days
        raw_capex = raw[..., 5:6]   # can be zero
        raw_stlt = raw[..., 6:7]    # fraction between 0 and 1

        # Constrain each driver to a financially meaningful domain
        price = tf.nn.softplus(raw_price) + 1e-3      # > 0
        volume = tf.nn.softplus(raw_volume) + 1e-3    # > 0
        
        # working capital levers in days; must be positive and not tiny
        dso = tf.nn.softplus(raw_dso) + 5.0            # min 5 days
        dpo = tf.nn.softplus(raw_dpo) + 5.0            # min 5 days
        dio = tf.nn.softplus(raw_dio) + 5.0            # min 5 days

        # capex as a non-negative cash outflow
        capex = tf.nn.softplus(raw_capex)              # can be zero

        # stlt_split in (0, 1): fraction of new debt that is LT debt
        stlt = tf.nn.sigmoid(raw_stlt)                 # fraction between 0 and 1

        return Drivers(
            price=price,
            volume=volume,
            dso_days=dso,
            dpo_days=dpo,
            dio_days=dio,
            capex=capex,
            stlt_split=stlt
        )

