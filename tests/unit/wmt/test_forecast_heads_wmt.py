import numpy as np
from wmt_bs_forecaster.forecast_heads import (
    avg_growth_next,
    ewma_next,
    BaselineHeads,
    BaselineHeadsConfig,
)


def test_avg_growth_next_simple():
    series = np.array([100.0, 110.0, 121.0], dtype=np.float32)  # +10% then +10%
    g, nxt = avg_growth_next(series, k=2)
    assert abs(g - 0.10) < 1e-6
    assert abs(nxt - 133.1) < 1e-3


def test_ewma_next_ratio():
    last = 0.20
    prev_fore = 0.18
    alpha = 0.4
    nxt = ewma_next(last, prev_fore, alpha)
    # next = alpha*last + (1-alpha)*prev = 0.4*0.20 + 0.6*0.18 = 0.188
    assert abs(nxt - 0.188) < 1e-6


def test_baseline_heads_fit_and_forecast():
    # Construct tiny history
    aoci = np.array([-12.0, -12.5, -12.9], dtype=np.float32)
    ni = np.array([7.0, 4.5, 5.3], dtype=np.float32)
    tr = np.array([-50.0, -55.0, -60.0], dtype=np.float32)  # becomes more negative
    rep = np.array([-1.0, -2.0, -3.0], dtype=np.float32)    # buyback outflow (negative)

    heads = BaselineHeads(BaselineHeadsConfig(alpha_opex=0.4, l2_aoci=0.1, l2_treasury=0.1))
    heads.fit({
        "aoci": aoci,
        "net_income": ni,
        "treasury_stock": tr,
        "repurchase_cash": rep,
    })

    last = {"sales": 121.0, "opex_ratio": 0.20, "aoci": aoci[-1], "treasury_stock": tr[-1]}
    prev_forecasts = {"opex_ratio": 0.19}
    features_next = {
        "sales_hist": np.array([100.0, 110.0, 121.0], dtype=np.float32),
        "net_income_next": 5.0,
        "repurchase_cash_next": -2.5,
    }
    out = heads.forecast_next(last, prev_forecasts, features_next)

    assert "sales_next" in out and out["sales_next"] > 0
    assert "opex_ratio_next" in out and 0.0 <= out["opex_ratio_next"] <= 1.0
    # AOCI and treasury forecasts may or may not be present depending on fit
    if "aoci_next" in out:
        assert np.isfinite(out["aoci_next"])  # numeric
    if "treasury_stock_next" in out:
        assert np.isfinite(out["treasury_stock_next"])  # numeric
