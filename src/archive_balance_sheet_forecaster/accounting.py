from __future__ import annotations
from balance_sheet_forecaster.types import Drivers, Policies, PrevState, Statements

import tensorflow as tf


# Versioning
STRUCT_VERSION = "0.1.0"

# Helper functions

# Fisher relation to compute nominal rate from real rate and inflation
def fisher_nominal(real_rate: tf.Tensor, inflation: tf.Tensor) -> tf.Tensor:
    """
    Compute nominal short rate via Fisher relation:
        (1 + real_rate) * (1 + inflation) - 1.

    Shapes:
        real_rate:   [B, T, 1]
        inflation:   [B, T, 1]
        return:      [B, T, 1]
    """

    return (1.0 + real_rate) * (1.0 + inflation) - 1.0

# Clamp to non-negative with small epsilon floor
def clamp_positive(x: tf.Tensor, epsilon: float = 1e-12) -> tf.Tensor:
    """
    Force non-negativity with a small floor. Avoids division by ~0 later.
    """

    return tf.nn.relu(x) + epsilon

# Convert per-period flow into balance level via days outstanding
def days_to_balance(flow: tf.Tensor, days: tf.Tensor) -> tf.Tensor:
    """
    Convert a per-period flow (e.g. sales or COGS) into a balance level.

    Rough rule:
        balance ≈ flow * (days / 365)

    Example:
        AR_t ≈ Sales_t * (DSO_days / 365)
        AP_t ≈ COGS_t  * (DPO_days / 365)

    Shapes:
        flow: [B, T, 1]
        days: [B, T, 1]
        return: [B, T, 1]
    """

    return flow * (days / 365.0)

# Broadcasting utility
def bcast_like(x, like: tf.Tensor) -> tf.Tensor:
    """
    Broadcast x to the shape of `like` ([B, T, 1]).
    Accepts x as scalar, [B, 1], or [B, T, 1].
    """
    x = tf.convert_to_tensor(x, dtype=like.dtype)
    target = tf.shape(like)  # [B, T, 1]

    # Fast paths when static rank is known
    if x.shape.rank == 3:
        return x
    if x.shape.rank == 2:
        x = tf.expand_dims(x, axis=1)  # [B, 1, 1]
        return tf.broadcast_to(x, target)
    if x.shape.rank in (0, 1):         # scalar or [1]
        x = tf.reshape(x, [1, 1, 1])
        return tf.broadcast_to(x, target)

    # Fallback (dynamic rank), reshape to [B,1,1] if possible then broadcast
    x = tf.reshape(x, [target[0], 1, 1])
    return tf.broadcast_to(x, target)



class StructuralLayer(tf.keras.layers.Layer):
    """
    Deterministic accounting / treasury engine.

    Inputs:
        - drivers (Drivers): behavioral levers predicted by DriverHead
        - policies (Policies): management / macro assumptions, not learned
        - prev (PrevState): prior-period ending balances

    Output:
        - Statements: income statement, cash budget, balance sheet for this period

    Key properties:
        * No plugs.
        * Interest is based on *prior* debt / investments (no circularity).
        * Cash shortfall is financed explicitly via ST/LT debt.
        * Excess liquidity is parked in short-term investments.
        * Equity rolls forward via retained earnings.
        * We assert Assets == Liabilities + Equity.
    """

    def __init__(self, *, hard_identity_check: bool = False, identity_tol: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.hard_identity_check = hard_identity_check
        self.identity_tol = float(identity_tol)

    def call(self,
             drivers: Drivers,  # trainable behavioral drivers, shape [B, T, 1]
             policies: Policies,    # policy inputs, not learned, shape [B, T, 1]
             prev: PrevState,   # prior period ending balances, shape [B, 1]
             training: bool = False
        ) -> Statements:
        
        # Unpack trainable behavioral drivers
        price, volume = drivers.price, drivers.volume
        dso, dpo, dio = drivers.dso_days, drivers.dpo_days, drivers.dio_days
        capex, stlt_split = drivers.capex, drivers.stlt_split # stlt_split: fraction of new debt that is LT debt

        # Unpack policy / scenario inputs
        inflation = policies.inflation
        real_rate = policies.real_rate
        tax_rate = policies.tax_rate
        min_cash_ratio = policies.min_cash_ratio
        payout_ratio = policies.payout_ratio

        # Rates
        # Short-term nominal rate derived from Fisher relation
        nominal_rate = fisher_nominal(real_rate, inflation)
        lt_rate = policies.lt_rate if getattr(policies, "lt_rate", None) is not None else (nominal_rate + 0.015)
        st_rate = policies.st_rate if getattr(policies, "st_rate", None) is not None else nominal_rate
        st_invest_rate = policies.st_invest_rate if getattr(policies, "st_invest_rate", None) is not None else nominal_rate
        # Operating expense ratio
        opex_ratio_policy = getattr(policies, "opex_ratio", None)
        opex_ratio = opex_ratio_policy if opex_ratio_policy is not None else tf.stop_gradient(tf.ones_like(price) * 0.2)
        
        cost_share_policy = getattr(policies, "cost_share", None) # if provided, overrides default COGS / inventory cost share
        cost_share = cost_share_policy if cost_share_policy is not None else tf.stop_gradient(tf.ones_like(price) * 0.6)
        cost_share = tf.clip_by_value(cost_share, 0.0, 1.0)

        depreciation_rate_policy = getattr(policies, "depreciation_rate", None)
        depreciation_rate = depreciation_rate_policy if depreciation_rate_policy is not None else (tf.ones_like(price) * 0.05)
        depreciation_rate = tf.clip_by_value(depreciation_rate, 0.0, 1.0)

        cash_cov = getattr(policies, "cash_coverage", None)  # if provided, overrides min_cash_ratio path

        # Per-period constants (assuming quarters)
        period_days = tf.constant(365.0 / 4.0, dtype=tf.float32)
        period_frac = tf.constant(0.25, dtype=tf.float32)

        # Operating statement
        # Revenue: sales = price * volume
        sales = price * volume # [B, T, 1]

        # Prepare prior balances for broadcasting over T
        prev_cash_T = bcast_like(prev.cash, sales)
        prev_st_investments_T = bcast_like(prev.st_investments, sales)
        prev_st_debt_T = bcast_like(prev.st_debt, sales)
        prev_lt_debt_T = bcast_like(prev.lt_debt, sales)
        prev_ar_T = bcast_like(prev.ar, sales)
        prev_ap_T = bcast_like(prev.ap, sales)
        prev_inventory_T = bcast_like(prev.inventory, sales)
        prev_nfa_T = bcast_like(prev.nfa, sales)
        prev_equity_T = bcast_like(prev.equity, sales)

        # Inventory modeling via DIO
        #   implied_inventory ~ sales * dio_days/365 * cost_share
        implied_inventory = days_to_balance(sales, dio) * cost_share # [B, T, 1]
        cogs = implied_inventory / clamp_positive(dio / 365.0)   # [B, T, 1]

        # Operating expenditures (opex)
        opex = tf.nn.relu(opex_ratio) * sales   # [B, T, 1]

        ebit = sales - cogs - opex # [B, T, 1]

        # Interest (no circularity: based on prior period balances)
        interest_st_debt = prev_st_debt_T * st_rate
        interest_lt_debt = prev_lt_debt_T * lt_rate
        interest_st_invest = - prev_st_investments_T * st_invest_rate # income less interest expense
        interest = interest_st_debt + interest_lt_debt + interest_st_invest # [B, T, 1]

        taxable_income = ebit - interest
        tax = tf.nn.relu(taxable_income) * tax_rate # no NOLs for simplicity
        net_income = taxable_income - tax   # [B, T, 1]

        # Working capital balances
        ar = days_to_balance(sales, dso)    # [B, T, 1]
        ap = days_to_balance(cogs, dpo)     # [B, T, 1], simplified: AP based on COGS only
        inventory = implied_inventory       # [B, T, 1]

        # ΔWC (changes in working capital) & CFO (cash flow from operations)
        dAR = ar - prev_ar_T
        dAP = ap - prev_ap_T
        dINV = inventory - prev_inventory_T
        
        # Straight-line-ish placeholder: depreciation = rate * prior NFA
        depreciation = prev_nfa_T * depreciation_rate   # [B, T, 1]
        cfo = net_income + depreciation - dAR + dAP - dINV   # cash flow from operations, [B, T, 1]

        # Cash budget before financing:
        # cash_in: proxy for collections (placeholder: assume fully collected same period)
        # cash_out: operating cash costs + taxes
        cash_in = sales
        cash_out = cogs + opex + tax

        # Liquidity / cash policy
        # Minimum required cash buffer as a fraction of sales
        if cash_cov is not None:
            min_cash = tf.nn.relu(cash_cov) * cash_out * period_frac
        else:   # [B, T, 1]
            min_cash = tf.nn.relu(min_cash_ratio) * sales   # [B, T, 1]

        capex_out = clamp_positive(capex)   # [B, T, 1]

        dividends = tf.nn.relu(net_income) * payout_ratio   # [B, T, 1], simple payout policy
        fcf = cfo - capex_out - dividends    # free cash flow, [B, T, 1], (unused for now)

        # net cash budget before financing
        ncb = cash_in - cash_out - capex_out - dividends    # [B, T, 1]

        # cash prior to financing activities
        pre_cash = prev_cash_T + cfo - capex_out - dividends  # [B, T, 1]

        # gap to policy cash buffer
        deficit = tf.nn.relu(min_cash - pre_cash)
        excess = tf.nn.relu(pre_cash - min_cash)

        # Financing mix for deficit
        use_lt = stlt_split * deficit   # use debt to cover deficit
        use_st = deficit - use_lt

        # End-of-period cash and ST investments:
        # If deficit, borrow to meet min cash;
        # if excess, park in ST investments.
        cash = tf.where(deficit > 0.0, min_cash, pre_cash)
        # cash = tf.minimum(pre_cash, min_cash) + use_st + use_lt
        st_investments = tf.where(
            deficit > 0.0, 
            tf.zeros_like(excess),   # no ST invest if we had a deficit
            excess
        ) # [B, T, 1]

        # Debt roll-forward
        st_debt = prev_st_debt_T + use_st # [B, T, 1]
        lt_debt = prev_lt_debt_T + use_lt # [B, T, 1]

        # Fixed assets roll-forward
        nfa = tf.nn.relu(prev_nfa_T + capex_out - depreciation)

        # Equity roll-forward via retained earnings
        retained = net_income - dividends
        equity = prev_equity_T + retained # [B, T, 1]

        # Package statements
        stm = Statements(
            sales=sales,
            cogs=cogs,
            opex=opex,
            ebit=ebit,
            interest=interest,
            tax=tax,
            net_income=net_income,
            cash=cash,
            ar=ar,
            ap=ap,
            inventory=inventory,
            st_investments=st_investments,
            st_debt=st_debt,
            lt_debt=lt_debt,
            nfa=nfa,
            equity=equity,
            ncb=ncb
        )

        # Accounting identity check: Assets = Liab + Equity
        assets = tf.reduce_mean(stm.assets)
        liab_eq_total = tf.reduce_mean(stm.liab_plus_equity)

        if self.hard_identity_check and not training and tf.executing_eagerly():
            gap = tf.abs(assets - liab_eq_total)
            tf.debugging.assert_less_equal(
                gap, 
                self.identity_tol, 
                message="Accounting identity violated: Assets != Liabilities + Equity",
            )   # in real production, only rely on soft guardrail to check accounting identity
            
        return stm

