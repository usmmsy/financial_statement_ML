from typing import Dict, Optional
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from .types_wmt import DriversWMT

# NOTE: This is a minimal, per-series ARIMA forecaster to emit DriversWMT paths.
# In practice you will likely precompute history from yfinance and then forecast T steps.

class ArimaDrivers:
    def __init__(self, order_map: Optional[Dict[str, tuple]] = None):
        # order_map like {"sales": (1,1,0), "cogs": (1,1,0), "dso_days": (1,0,0), ...}
        self.order_map = order_map or {}

    def _fit_forecast(self, y_hist: np.ndarray, steps: int, order=(1,1,0)) -> np.ndarray:
        y_hist = np.asarray(y_hist, dtype=float)
        if len(y_hist) < max(8, sum(order) + 2):
            # fallback: naive last
            return np.repeat(float(y_hist[-1]), steps)
        model = ARIMA(y_hist, order=order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(method_kwargs={"warn_convergence": False})
        fc = res.forecast(steps=steps)
        return np.asarray(fc, dtype=float)

    def forecast(self,
                 hist: Dict[str, np.ndarray],
                 steps: int,
                 opex_ratio: float = 0.18,
                 depreciation_rate: float = 0.10,
                 capex: Optional[np.ndarray] = None) -> DriversWMT:
        """hist keys expected: sales, cogs, dso_days, dpo_days, dio_days (each [N])"""
        get = lambda k: np.asarray(hist[k], dtype=float)
        ord_for = lambda k, dflt: self.order_map.get(k, dflt)

        sales_fc = self._fit_forecast(get("sales"), steps, ord_for("sales", (1,1,0)))
        cogs_fc  = self._fit_forecast(get("cogs"),  steps, ord_for("cogs",  (1,1,0)))

        dso_fc = np.maximum(self._fit_forecast(get("dso_days"), steps, ord_for("dso_days", (1,0,0))), 0.0)
        dpo_fc = np.maximum(self._fit_forecast(get("dpo_days"), steps, ord_for("dpo_days", (1,0,0))), 0.0)
        dio_fc = np.maximum(self._fit_forecast(get("dio_days"), steps, ord_for("dio_days", (1,0,0))), 0.0)

        if capex is not None:
            capex = np.asarray(capex, dtype=float)
            if len(capex) != steps:
                raise ValueError("capex length must equal steps when provided")

        # Convert to [B=1, T, 1] tensors
        def t3(x: np.ndarray) -> tf.Tensor:
            x = np.asarray(x, dtype=np.float32).reshape(1, steps, 1)
            return tf.convert_to_tensor(x)

        capex_tf = None if capex is None else t3(capex)

        return DriversWMT(
            sales=t3(sales_fc),
            cogs=t3(cogs_fc),
            dso_days=t3(dso_fc),
            dpo_days=t3(dpo_fc),
            dio_days=t3(dio_fc),
            opex_ratio=t3(np.full((steps,), float(opex_ratio), dtype=np.float32)),
            capex=capex_tf,
            depreciation_rate=t3(np.full((steps,), float(depreciation_rate), dtype=np.float32)),
        )