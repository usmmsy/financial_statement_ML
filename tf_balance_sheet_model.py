
from typing import Optional, Dict, Tuple
import numpy as np
import tensorflow as tf

Tensor = tf.Tensor

def fisher_nominal_series(real_rate: Tensor, inflation: Tensor) -> Tensor:
    """
    Per-period Fisher: r_nom_t = (1+r_real_t)*(1+pi_t)-1
    Shapes:
      real_rate: [T] or [B,T]
      inflation: [T] or [B,T]
    Returns: same shape as broadcast(real_rate, inflation)
    """
    return (1.0 + tf.nn.relu(real_rate)) * (1.0 + inflation) - 1.0

def build_price_path(p0: Tensor, inflation: Tensor) -> Tensor:
    """
    Price path using per-period inflation vector.
    p_t = p0 * prod_{k=0..t-1} (1+pi_k)
    Inputs:
      p0: scalar or [B,1]
      inflation: [T] or [B,T]
    Output:
      prices: [T] or [B,T]
    """
    one = tf.ones_like(inflation)
    gross = one + inflation
    cum = tf.math.cumprod(gross, axis=-1, exclusive=False)
    return p0 * cum

def build_cost_path(c0: Tensor, inflation_cost: Tensor) -> Tensor:
    """
    Similar to price path, but allow different inflation for costs.
    """
    one = tf.ones_like(inflation_cost)
    gross = one + inflation_cost
    cum = tf.math.cumprod(gross, axis=-1, exclusive=False)
    return c0 * cum

def compute_inventory_units(units: Tensor,
                            inv_months: int,
                            months_in_year: int = 12,
                            inv_units_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Paper policy: EndInvUnits_t = (inv_months/months_in_year) * Units_{t+1}
    We precompute beg, end, and purchases units series from the sales units path.

    Args:
      units: [T] or [B,T] sales units for each period.
      inv_months: policy months of next year's volume to stock (e.g., 1 for 1/12).
      months_in_year: typically 12.
      inv_units_0: initial beginning inventory units at t=0. If None, set to policy * Units_0.

    Returns:
      inv_beg_units: [T]
      inv_end_units: [T]
      purchases_units: [T] = inv_end + units - inv_beg
    """
    frac = tf.cast(inv_months, units.dtype) / float(months_in_year)
    last = units[..., -1:]
    units_next = tf.concat([units[..., 1:], last], axis=-1)
    inv_end = frac * units_next
    if inv_units_0 is None:
        inv_beg0 = frac * units[..., :1]
    else:
        inv_beg0 = tf.convert_to_tensor(inv_units_0, dtype=units.dtype)
    inv_beg = tf.concat([inv_beg0, inv_end[..., :-1]], axis=-1)
    purchases_units = inv_end + units - inv_beg
    return inv_beg, inv_end, purchases_units

def fifo_current_cost_valuation(units: Tensor,
                                cost_per_unit: Tensor,
                                inv_beg_units: Tensor,
                                inv_end_units: Tensor,
                                purchases_units: Tensor,
                                purchases_unit_cost: Tensor) -> Tuple[Tensor, Tensor]:
    """
    FIFO valuation at current cost (paper's simplification):
      COGS_t = UnitsSold_t * current unit cost_t
      EndInvValue_t = inv_end_units_t * current unit cost_t
    Although called FIFO, the paper example values inventories at current period costs.
    Returns:
      cogs_value_t, end_inventory_value_t
    """
    cogs = units * cost_per_unit
    inv_end_value = inv_end_units * cost_per_unit
    return cogs, inv_end_value

def default_lt_amort_schedule(principal0: float, T: int, years: int = 10):
    """
    Creates a simple equal-principal amortization schedule over `years` (clipped to horizon T).
    Returns an array of shape [T] with principal outflows per period, starting at t=1.
    """
    sched = np.zeros(T, dtype=np.float32)
    if principal0 <= 0 or years <= 0:
        return sched
    term = min(T-1, years)
    per = principal0 / term if term > 0 else 0.0
    if term > 0:
        sched[1:1+term] = per
    return sched

def run_model(
    T: int,
    units: Tensor,
    p0: Tensor,
    c0: Tensor,
    inflation_price: Tensor,
    inflation_cost: Tensor,
    commission_pct_sales: float,
    advertising_pct_sales: float,
    overhead0: Tensor,
    payroll0: Tensor,
    overhead_infl: Tensor,
    payroll_infl: Tensor,
    ar_days: float,
    ap_days: float,
    min_cash_pct_sales: float,
    inv_months: int,
    rf_real: Tensor,
    st_spread: float,
    st_invest_spread: float,
    lt_rate: float,
    inv_units_0: Optional[Tensor] = None,
    lt_principal0: float = 0.0,
    lt_principal_sched: Optional[Tensor] = None,
    tax_rate: float = 0.25,
    payout_ratio: float = 0.0,
) -> Dict[str, Tensor]:
    """
    Paper-faithful deterministic engine.
    Returns a dict of series tensors.
    """
    def ensure_bt(x, B=None):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if len(x.shape) == 0:
            x = tf.reshape(x, [1, 1])
        elif len(x.shape) == 1:
            x = tf.expand_dims(x, 0)
        if B is not None and tf.shape(x)[0] == 1 and B > 1:
            x = tf.tile(x, [B, 1])
        return x

    units = tf.convert_to_tensor(units, dtype=tf.float32)
    if len(units.shape) == 1:
        units = tf.expand_dims(units, 0)  # [1,T]
    B = tf.shape(units)[0]

    rf_real = ensure_bt(rf_real, B)
    inflation_price = ensure_bt(inflation_price, B)
    inflation_cost = ensure_bt(inflation_cost, B)
    overhead_infl = ensure_bt(overhead_infl, B)
    payroll_infl = ensure_bt(payroll_infl, B)

    rf_nom = fisher_nominal_series(rf_real, inflation_price)

    prices = build_price_path(ensure_bt(p0, B), inflation_price)  # [B,T]
    unit_costs = build_cost_path(ensure_bt(c0, B), inflation_cost)

    overhead_path = build_cost_path(ensure_bt(overhead0, B), overhead_infl)
    payroll_path = build_cost_path(ensure_bt(payroll0, B), payroll_infl)

    inv_beg_u, inv_end_u, purchases_u = compute_inventory_units(units, inv_months, 12, inv_units_0)
    cogs_val, inv_end_val = fifo_current_cost_valuation(
        units=units,
        cost_per_unit=unit_costs,
        inv_beg_units=inv_beg_u,
        inv_end_units=inv_end_u,
        purchases_units=purchases_u,
        purchases_unit_cost=unit_costs
    )

    sales = units * prices
    revenue = tf.reduce_sum(sales, axis=0) if tf.shape(sales)[0] > 1 else tf.squeeze(sales, axis=0)
    cogs = tf.reduce_sum(cogs_val, axis=0) if tf.shape(cogs_val)[0] > 1 else tf.squeeze(cogs_val, axis=0)

    commissions = commission_pct_sales * revenue
    advertising = advertising_pct_sales * revenue
    overhead = tf.reduce_sum(overhead_path, axis=0) if tf.shape(overhead_path)[0] > 1 else tf.squeeze(overhead_path, axis=0)
    payroll = tf.reduce_sum(payroll_path, axis=0) if tf.shape(payroll_path)[0] > 1 else tf.squeeze(payroll_path, axis=0)

    as_total = commissions + advertising + overhead + payroll

    gross_profit = revenue - cogs
    ebit = gross_profit - as_total

    days_in_year = 365.0
    ar = revenue * (ar_days / days_in_year)
    ap = cogs * (ap_days / days_in_year)
    min_cash = min_cash_pct_sales * revenue

    rf_borrow = rf_nom + st_spread
    rf_invest = tf.nn.relu(rf_nom - st_invest_spread)

    if lt_principal_sched is None:
        lt_principal_sched_np = default_lt_amort_schedule(lt_principal0, T, years=10)
        lt_principal_sched = tf.convert_to_tensor(lt_principal_sched_np, dtype=tf.float32)
    else:
        lt_principal_sched = tf.convert_to_tensor(lt_principal_sched, dtype=tf.float32)

    def scan_step(carry, t):
        st_bb = carry["st_bb"]
        st_principal_due = carry["st_principal_due"]
        lt_bb = carry["lt_bb"]
        cash_beg = carry["cash_beg"]
        retained_earnings = carry["ret_earn"]

        rev_t = revenue[t]
        ebit_t = ebit[t]
        ar_t = ar[t]
        ap_t = ap[t]
        min_cash_t = min_cash[t]

        rf_borrow_t = rf_borrow[..., t] if len(rf_borrow.shape) > 1 else rf_borrow[t]
        rf_invest_t = rf_invest[..., t] if len(rf_invest.shape) > 1 else rf_invest[t]

        st_int = st_bb * rf_borrow_t
        lt_int = lt_bb * lt_rate

        ebt_t = ebit_t - st_int - lt_int
        tax_t = tf.nn.relu(tax_rate * ebt_t)
        ni_t = ebt_t - tax_t
        div_t = payout_ratio * tf.nn.relu(carry["ni_prev"])

        d_ar = ar_t - carry["ar_prev"]
        d_ap = ap_t - carry["ap_prev"]
        ocf = (ebit_t - tax_t) + ( - d_ar + d_ap )

        lt_prin_t = lt_principal_sched[t] if t < tf.shape(lt_principal_sched)[0] else 0.0
        mandatory_uses = div_t + lt_prin_t + st_principal_due
        cash_pre = cash_beg + ocf - mandatory_uses

        gap = min_cash_t - cash_pre
        st_new = tf.nn.relu(gap)
        surplus = tf.nn.relu(cash_pre - min_cash_t)
        cash_end = cash_pre + st_new
        invest_income = surplus * rf_invest_t

        st_eb = st_bb + st_new - st_principal_due
        lt_eb = tf.nn.relu(lt_bb - lt_prin_t)
        ret_earn_next = retained_earnings + (ni_t - div_t)

        out = {
            "EBIT": ebit_t,
            "EBT": ebt_t,
            "Tax": tax_t,
            "NI": ni_t,
            "Div": div_t,
            "OCF": ocf,
            "LT_principal": lt_prin_t,
            "ST_interest": st_int,
            "LT_interest": lt_int,
            "Invest_income": invest_income,
            "Cash_end": cash_end,
            "ST_new": st_new,
            "ST_principal_due_next": st_new,
            "ST_bb": st_bb,
            "ST_eb": st_eb,
            "LT_bb": lt_bb,
            "LT_eb": lt_eb,
        }

        carry_next = {
            "st_bb": st_eb,
            "st_principal_due": out["ST_principal_due_next"],
            "lt_bb": lt_eb,
            "cash_beg": cash_end,
            "ret_earn": ret_earn_next,
            "ni_prev": ni_t,
            "ar_prev": ar_t,
            "ap_prev": ap_t,
        }
        return carry_next, out

    init_carry = {
        "st_bb": tf.constant(0.0, dtype=tf.float32),
        "st_principal_due": tf.constant(0.0, dtype=tf.float32),
        "lt_bb": tf.constant(lt_principal0, dtype=tf.float32),
        "cash_beg": tf.constant(0.0, dtype=tf.float32),
        "ret_earn": tf.constant(0.0, dtype=tf.float32),
        "ni_prev": tf.constant(0.0, dtype=tf.float32),
        "ar_prev": ar[0] if len(ar.shape) > 0 else ar,
        "ap_prev": ap[0] if len(ap.shape) > 0 else ap,
    }

    _, outs = tf.scan(scan_step,
                      elems=tf.range(T),
                      initializer=init_carry,
                      parallel_iterations=1)

    cash_end = outs["Cash_end"]
    mct = min_cash
    cum_ncb = tf.math.cumsum(cash_end - mct)

    result = {
        "Revenue": revenue,
        "COGS": cogs,
        "GrossProfit": revenue - cogs,
        "Commissions": commissions,
        "Advertising": advertising,
        "Overhead": overhead,
        "Payroll": payroll,
        "EBIT": outs["EBIT"],
        "EBT": outs["EBT"],
        "Tax": outs["Tax"],
        "NI": outs["NI"],
        "Dividends": outs["Div"],
        "OCF": outs["OCF"],
        "ST_Interest": outs["ST_interest"],
        "LT_Interest": outs["LT_interest"],
        "InvestIncome": outs["Invest_income"],
        "CashEnd": cash_end,
        "ST_New": outs["ST_new"],
        "ST_BegBal": outs["ST_bb"],
        "ST_EndBal": outs["ST_eb"],
        "LT_BegBal": outs["LT_bb"],
        "LT_EndBal": outs["LT_eb"],
        "InvEndUnits": tf.reduce_sum(inv_end_u, axis=0) if len(inv_end_u.shape) == 2 else inv_end_u,
        "InvEndValue": tf.reduce_sum(inv_end_val, axis=0) if len(inv_end_val.shape) == 2 else inv_end_val,
        "CumNCB_minus_MCT": cum_ncb,
        "MinCashTarget": mct,
        "Prices": tf.reduce_sum(prices, axis=0) if len(prices.shape) == 2 else prices,
        "UnitCosts": tf.reduce_sum(unit_costs, axis=0) if len(unit_costs.shape) == 2 else unit_costs,
        "RF_nom": tf.reduce_sum(rf_nom, axis=0) if len(rf_nom.shape) == 2 else rf_nom,
    }
    return result
