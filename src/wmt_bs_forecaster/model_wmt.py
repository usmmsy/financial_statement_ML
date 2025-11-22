from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from .types_wmt import PoliciesWMT, DriversWMT, PrevStateWMT, StatementsWMT
from .accounting_wmt import StructuralLayer

class WalmartBSModel:
    def __init__(self, policies: PoliciesWMT, structural: Optional[StructuralLayer] = None):
        self.policies = policies
        self.structural = structural or StructuralLayer(policies)
        self.driver_model = None

    def attach_arima_driver(self, order_map: Optional[Dict[str, tuple]] = None):
        from .drivers_wmt import ArimaDrivers  # optional dependency
        self.driver_model = ArimaDrivers(order_map)

    def forecast_with_given_drivers(self, drivers: DriversWMT, prev: PrevStateWMT) -> StatementsWMT:
        # TF-native path: return tensors [B, T, 1]
        return self.structural.call(drivers, prev)

    def forecast_with_arima(self,
                             history: Dict[str, np.ndarray],
                             prev: PrevStateWMT,
                             steps: int,
                             opex_ratio: float = 0.18,
                             depreciation_rate: float = 0.10,
                             capex: Optional[np.ndarray] = None) -> StatementsWMT:
        if self.driver_model is None:
            from .drivers_wmt import ArimaDrivers  # optional dependency
            self.driver_model = ArimaDrivers()
        drivers = self.driver_model.forecast(
            hist=history,
            steps=steps,
            opex_ratio=opex_ratio,
            depreciation_rate=depreciation_rate,
            capex=capex,
        )
        return self.structural.call(drivers, prev)

    def rollout(self, drivers: DriversWMT, prev: PrevStateWMT) -> Tuple[StatementsWMT, PrevStateWMT]:
        """Run the structural layer and return statements plus final prev-state for chaining.

        This enables sequential multi-horizon inference: you can feed the returned PrevStateWMT
        into a subsequent forecast call with new drivers/policies.
        """
        stm = self.structural.call(drivers, prev)
        # Extract final slice (t = T-1) to form next PrevStateWMT
        def last(x: tf.Tensor) -> tf.Tensor:
            return x[:, -1:, :] if x.shape.rank == 3 else x  # keep [B,1,1]

        next_prev = PrevStateWMT(
            cash=last(stm.cash),
            st_investments=last(stm.st_investments),
            st_debt=last(stm.st_debt),
            lt_debt=last(stm.lt_debt),
            ar=last(stm.ar),
            ap=last(stm.ap),
            inventory=last(stm.inventory),
            net_ppe=last(stm.net_ppe),
            equity=last(stm.equity),
            retained_earnings=last(stm.retained_earnings),
            paid_in_capital=last(stm.paid_in_capital),
        )
        return stm, next_prev