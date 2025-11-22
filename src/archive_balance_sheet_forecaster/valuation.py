from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal, Union

import numpy as np
import tensorflow as tf

ArrayLike = Union[np.ndarray, tf.Tensor, float]

def _to_tf(x: ArrayLike) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return x
    return tf.convert_to_tensor(x, dtype=tf.float32)

def _pv(cash: ArrayLike, rates: ArrayLike) -> tf.Tensor:
    """
    Present value of a per-period cashflow sequence at per-period rates.

    cash:  [T]  cash[t] pays at end of period t (t=1..T interpreted via index 0..T-1)
    rates: [T]  rate[t] applicable for discounting cash[t]
    """
    c = _to_tf(cash)         # [T]
    r = _to_tf(rates)        # [T]
    # cumprod(1+r): [1+r1, (1+r1)(1+r2), ...]
    disc = tf.math.cumprod(1.0 + r)
    return tf.reduce_sum(c / disc)

def _levelize(x: ArrayLike, T: int) -> tf.Tensor:
    """
    Broadcast a scalar or [T] vector to [T].
    """
    t = _to_tf(x)
    if t.shape.rank == 0:
        return tf.ones([T], dtype=tf.float32) * t
    return t

@dataclass
class ValuationStreams:
    """
    Minimal streams you need to run non-iterative valuation.

    Time index convention: arrays are length T and represent cash at the *end*
    of periods 1..T (index 0..T-1). All rates are per-period effective rates.
    """
    # Core operating flows from your structural engine:
    # FCF to the firm (after-tax operating CF to all capital providers, *excluding* tax shields)
    fcf: ArrayLike                        # [T]
    # Interest expense and tax rate arrays (used to build tax shields if not given)
    interest: Optional[ArrayLike] = None  # [T] interest expense each period
    tax_rate: Optional[ArrayLike] = None  # [T] effective tax rate each period

    # Optional direct tax shield series. If None, we’ll compute τ * interest.
    tax_shield: Optional[ArrayLike] = None  # [T]

    # Optional debt cash-flow view if you want a market value of debt:
    # coupons (interest paid) and principal changes/redemptions (negative for issuance, positive for repayment)
    debt_coupons: Optional[ArrayLike] = None   # [T]
    debt_principal_cf: Optional[ArrayLike] = None  # [T] (e.g., +repayment outflow to debt holders)

@dataclass
class ValuationAssumptions:
    """
    Discount-rate assumptions and leverage knobs.

    Ku: unlevered cost of capital (can be scalar or [T])
    Kd: cost of debt (can be scalar or [T])
    tau_for_wacc: tax rate for WACC computation (scalar or [T]; if None we’ll use
                  average of streams.tax_rate when computing WACC)
    """
    Ku: ArrayLike                      # scalar or [T]
    Kd: Optional[ArrayLike] = None     # scalar or [T]; needed if you want market D or APV split
    tau_for_wacc: Optional[ArrayLike] = None

@dataclass
class ValuationResult:
    """
    Outputs from non-iterative valuation.
    """
    V: float                # levered firm value (at t0)
    D: Optional[float]      # market value of debt if computed (else None)
    E: float                # equity value (= V - D if D provided; else V - book D you pass in)
    Ke: Optional[float]     # implied cost of equity (simple static formula), optional
    WACC: Optional[float]   # simple WACC (period-0 weights), optional
    method: Literal["CCF", "APV"]

# ----------------------------
# Builders for “no-iteration” valuation
# ----------------------------

def build_ccf(fcf: ArrayLike, tax_shield: Optional[ArrayLike], interest: Optional[ArrayLike], tax_rate: Optional[ArrayLike]) -> tf.Tensor:
    """
    Capital Cash Flow (Ruback-style): CCF = FCF + Tax Shield.
    If tax_shield is None, we use τ * interest.
    """
    fcf_t = _to_tf(fcf)               # [T]
    if tax_shield is not None:
        ts = _to_tf(tax_shield)       # [T]
    else:
        assert interest is not None and tax_rate is not None, \
            "Provide either tax_shield or (interest and tax_rate) to build CCF."
        ts = _to_tf(interest) * _to_tf(tax_rate)  # [T]
    return fcf_t + ts                  # [T]

def value_via_ccf(streams: ValuationStreams, assump: ValuationAssumptions) -> ValuationResult:
    """
    Non-iterative levered firm value: PV(CCF) discounted at Ku.
    Then equity = V - D. If you want D (market), provide debt CFs & Kd;
    else pass an external D_book and we’ll subtract that outside.
    """
    # Build CCF
    T = int(_to_tf(streams.fcf).shape[0])
    Ku = _levelize(assump.Ku, T)      # [T]
    ccf = build_ccf(streams.fcf, streams.tax_shield, streams.interest, streams.tax_rate)  # [T]

    V = float(_pv(ccf, Ku).numpy())   # levered firm value

    # Optional market value of debt from debt-holder CFs discounted at Kd
    D_mkt = None
    if streams.debt_coupons is not None and streams.debt_principal_cf is not None and assump.Kd is not None:
        Kd = _levelize(assump.Kd, T)
        debt_cf = _to_tf(streams.debt_coupons) + _to_tf(streams.debt_principal_cf)   # [T]
        D_mkt = float(_pv(debt_cf, Kd).numpy())

    # Return Ke/WACC only if we have a notion of capital weights now
    Ke, WACC = None, None
    if D_mkt is not None:
        E = V - D_mkt
        if E <= 0.0:
            return ValuationResult(V=V, D=D_mkt, E=E, Ke=None, WACC=None, method="CCF")
        # Simple static (period-0) weights & tax for headline numbers
        w_d = D_mkt / V
        w_e = E / V
        tau = _to_tf(streams.tax_rate) if assump.tau_for_wacc is None else _to_tf(assump.tau_for_wacc)
        tau0 = float(tf.reduce_mean(_levelize(tau, T)).numpy())
        Ku0 = float(_levelize(assump.Ku, T)[0].numpy())
        Kd0 = float(_levelize(assump.Kd, T)[0].numpy())
        # A common static relationship (no iteration, one-period weights)
        Ke = Ku0 + (Ku0 - Kd0) * (D_mkt / E)   # tax already handled by CCF approach in V
        WACC = w_e * Ke + w_d * Kd0 * (1.0 - tau0)
        return ValuationResult(V=V, D=D_mkt, E=E, Ke=Ke, WACC=WACC, method="CCF")

    # If we didn’t compute D here, leave it None and let caller subtract book/externally supplied D
    return ValuationResult(V=V, D=None, E=V, Ke=None, WACC=None, method="CCF")

def value_via_apv(streams: ValuationStreams, assump: ValuationAssumptions) -> ValuationResult:
    """
    APV (Adjusted Present Value), still non-iterative:
      V = PV(FCF discounted at Ku) + PV(Tax Shields discounted at chosen rate)
    Default here: discount tax shields at Ku as a simple, conservative stance.
    You can change the shield discount curve if you want Kd for shields.
    """
    T = int(_to_tf(streams.fcf).shape[0])
    Ku = _levelize(assump.Ku, T)

    # Unlevered value
    Vu = float(_pv(streams.fcf, Ku).numpy())

    # Tax shield stream
    ts = streams.tax_shield if streams.tax_shield is not None else _to_tf(streams.interest) * _to_tf(streams.tax_rate)
    ts = _to_tf(ts)

    # Choose a discount curve for shields: here use Ku by default
    Vts = float(_pv(ts, Ku).numpy())

    V = Vu + Vts

    D_mkt = None
    if streams.debt_coupons is not None and streams.debt_principal_cf is not None and assump.Kd is not None:
        Kd = _levelize(assump.Kd, T)
        debt_cf = _to_tf(streams.debt_coupons) + _to_tf(streams.debt_principal_cf)
        D_mkt = float(_pv(debt_cf, Kd).numpy())

    if D_mkt is not None:
        E = V - D_mkt
        # Optional static Ke/WACC headline
        if E > 0.0:
            w_d = D_mkt / V
            w_e = E / V
            tau = _to_tf(streams.tax_rate) if assump.tau_for_wacc is None else _to_tf(assump.tau_for_wacc)
            tau0 = float(tf.reduce_mean(_levelize(tau, T)).numpy())
            Ku0 = float(_levelize(assump.Ku, T)[0].numpy())
            Kd0 = float(_levelize(assump.Kd, T)[0].numpy())
            Ke = Ku0 + (Ku0 - Kd0) * (D_mkt / E)
            WACC = w_e * Ke + w_d * Kd0 * (1.0 - tau0)
        else:
            Ke, WACC = None, None
        return ValuationResult(V=V, D=D_mkt, E=E, Ke=Ke, WACC=WACC, method="APV")

    return ValuationResult(V=V, D=None, E=V, Ke=None, WACC=None, method="APV")
