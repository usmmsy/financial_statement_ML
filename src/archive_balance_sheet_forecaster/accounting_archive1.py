from __future__ import annotations
from typing import Tuple, Optional, List
from balance_sheet_forecaster import drivers
import tensorflow as tf

from balance_sheet_forecaster.types import (
    Learnables, Statements, Policies, PrevState,
)

def fisher_nominal(real_rate: tf.Tensor, inflation: tf.Tensor) -> tf.Tensor:
    # (1+r_nom) = (1+r_real)*(1+pi) - 1
    return (1.0 + real_rate) * (1.0 + inflation) - 1.0

def at_t(x: tf.Tensor) -> tf.Tensor:
    """Gather [B,T,1] -> [B,1,1] at current t. Assumes x is time-major on axis=1."""
    return tf.gather(x, t, axis=1) if x is not None else None

def bcast_like(x: tf.Tensor, ref: tf.Tensor) -> tf.Tensor:
    # expand [B,1,1] to [B,T,1] by tiling to ref shape
    tiles = tf.concat([tf.ones_like(ref[:, :1, :], dtype=tf.int32) * tf.shape(ref)[0:1],
                       tf.ones_like(ref[:, :1, :], dtype=tf.int32) * tf.shape(ref)[1:2],
                       tf.ones_like(ref[:, :1, :], dtype=tf.int32) * tf.shape(ref)[2:3]], axis=0)
    # Simpler and stable: just tile along time
    return tf.tile(x, [1, tf.shape(ref)[1], 1])

def forecast_volume_price_sales_step(
        elasticity_b: tf.Tensor,               # [B,1,1]
        elasticity_b0: tf.Tensor,           # [B,1,1]
        base_selling_price: tf.Tensor,    # [B,1,1]
        volume_growth_t: tf.Tensor,      # [B,1,1]
        prev_units: tf.Tensor,           # [B,1,1]
        prev_price: tf.Tensor,           # [B,1,1]
        nom_inc_sell_t: tf.Tensor,      # [B,1,1]
        is_initial: tf.Tensor,              # scalar bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Table 4: Forecast volume, price, sales (single period):
      units_t  = elasticity_b0 * (base_selling_price)^(elasticity_b)          if is_initial
                 prev_units * volume_growth_t                                 else
        price_t  = base_selling_price                                          if is_initial
                    prev_price * (1 + nom_inc_sell_t)                           else
        sales_t  = units_t * price_t
    """
    if is_initial:
        units_t = elasticity_b0 * tf.pow(base_selling_price, elasticity_b)  # [B, 1, 1]
        price_t = base_selling_price  # [B, 1, 1]
    else:
        units_t = prev_units * volume_growth_t  # [B, 1, 1]
        price_t = prev_price * (1 + nom_inc_sell_t)  # [B, 1, 1]
        sales_t = units_t * price_t  # [B, 1, 1]
    return units_t, price_t, sales_t

def capex_depr_step_capped_L(
    prev_nfa: tf.Tensor,                  # [B,1,1]
    growth_cap_yrs: tf.Tensor,            # [B,1,1] int32
    volume_growth_t: tf.Tensor,           # [B,1,1]
    depr_life_t: tf.Tensor,               # [B,1,1] int32 or float; e.g., 4
    init_fa: tf.Tensor,                   # [B,1,1]
    is_initial: tf.Tensor,                # scalar bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Table 5 (one period) with lineal life L:
      r = 1/L
      Dep_t            = r * prev_nfa
      Capex_repl_t     = prev_last_depr           (t=0 special: = prev_nfa)
      Capex_growth_t   = prev_nfa * volume_growth_t if growth_years_done < L else 0
      Capex_t          = Capex_repl_t + Capex_growth_t
      NFA_end_t        = prev_nfa + Capex_t - Dep_t
      next_last_depr   = Dep_t
      next_growth_done = growth_years_done + 1 if Capex_growth_t > 0
    """
    # cast/derive r = 1/L
    L_float = tf.cast(depr_life_t, prev_nfa.dtype)                 # [B,1,1]
    r = 1.0 / tf.maximum(L_float, 1.0)                               # guard L>=1

    # depreciation (straight-line on opening stock)
    dep_t = r * prev_nfa                                            # [B,1,1], 0 at t=0 since nfa = 0 at t=0

    # replacement capex / investment to maintain existing asset base
    capex_repl_t = dep_t                                # [B,1,1], 0 at t=0 since depr = 0 at t=0

    # growth capex with hard cap after L years / investment in fixed assets for growth
    life_int = tf.cast(tf.round(L_float), tf.int32)                 # [B,1,1]
    can_grow = tf.less(growth_cap_yrs, life_int)                    # [B,1,1] bool
    capex_growth_raw = prev_nfa * volume_growth_t                   # [B,1,1]
    capex_growth_t   = tf.where(can_grow, capex_growth_raw, tf.zeros_like(capex_growth_raw))

    # totals / roll
    capex_t  = capex_repl_t + capex_growth_t                        # [B,1,1]
    
    # [B,1,1], NFA rollforward: NFA_end = NFA_start + (Capex_growth + Capex_replace) - Depr
    nfa_end_t = prev_nfa + capex_t - dep_t    

    did_grow = tf.greater(capex_growth_t, 0.0)                      # [B,1,1] bool
    growth_done_next = tf.where(
        did_grow,
        growth_cap_yrs + tf.ones_like(growth_cap_yrs),
        growth_cap_yrs
    )
    last_depr_next = dep_t

    # initial-year conventions
    nfa_end_t         = tf.where(is_initial, init_fa, nfa_end_t)

    return dep_t, capex_repl_t, capex_growth_t, capex_t, nfa_end_t, last_depr_next, growth_done_next

def inventory_purchases_units_step(
    units_sold_t: tf.Tensor,              # [B,1,1]
    prev_inv_units: tf.Tensor,          # [B,1,1]
    inventory_volume_ratio: tf.Tensor,     # [B,1,1]
    is_initial: tf.Tensor,                # scalar bool
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Table 6a: Inventory and purchases in units (single period):
    Returns:
      final_inv_units_t  [B,1,1]  (Table 6a row 82)
      purchases_units_t  [B,1,1]  (Table 6a row 84)
    """
    target_final_inv = units_sold_t * inventory_volume_ratio  # [B,1,1]
    final_inv_units_t = tf.where(
        is_initial,
        prev_inv_units,
        target_final_inv
    )  # [B,1,1]

    purchase_units_t = tf.where(
        is_initial,
        final_inv_units_t,
        units_sold_t + final_inv_units_t - prev_inv_units,  # [B,1,1]
    )  # [B,1,1]
    
    return final_inv_units_t, purchase_units_t

def inventory_valuation_fifo_cogs_step(
        prev_inv_value: tf.Tensor,        # [B,1,1]
        unit_cost: tf.Tensor,             # [B,1,1]
        purchases_units_t: tf.Tensor,      # [B,1,1]
        final_inv_units_t: tf.Tensor,    # [B,1,1]
        is_initial: tf.Tensor,            # scalar bool
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Table 6b: Inventory valuation using FIFO and COGS in $ (single period):
    Returns:
      inv_unit_cost_t  [B,1,1]  (Table 6b row 86)
      cogs_t          [B,1,1]  (Table 6b row 88)
    """
    if is_initial:
        unit_cost_t = unit_cost  # [B,1,1]
        cogs_t = tf.zeros_like(unit_cost)  # [B,1,1]
    purchases_value_t = purchases_units_t * unit_cost  # [B,1,1]
    final_inv_value_t = final_inv_units_t * unit_cost  # [B,1,1]

class StructuralLayer(tf.keras.layers.Layer):
    def __init__(self, name: str = "structural_layer"):
        super().__init__(name=name)
        # Any parameters or sub-layers can be defined here if needed

    def call(
        self, 
        learnables: Learnables, 
        policies: Policies, 
        prev: PrevState, 
        training: bool = False,
    ) -> Statements:
        
        # === Policies / learnables at time t ===

        # -- time index & helper --
        t = tf.cast(prev.t, tf.int32)           # scalar tensor
        is_initial = tf.equal(t, 0)             # scalar bool tensor

        inflation = at_t(learnables.inflation)          # [B, 1, 1]
        real_rate = at_t(learnables.real_rate)          # [B, 1, 1]
        risk_premium_debt = at_t(learnables.risk_premium_debt)          # [B, 1, 1]
        risk_premium_st_invest = at_t(learnables.risk_premium_st_invest)          # [B, 1, 1]

        # Real (above-inflation) drifts (Table 1a)
        real_increase_sell_price = at_t(learnables.real_increase_sell_price)  # [B, 1, 1] row 19  real increase in selling price
        real_increase_unit_cost = at_t(learnables.real_increase_unit_cost)  # [B, 1, 1] row 20  real increase in purchase/unit cost
        real_increase_overhead = at_t(learnables.real_increase_overhead)  # [B, 1, 1] row 21  real increase in overhead expenses
        real_increase_payroll = at_t(learnables.real_increase_payroll)  # [B, 1, 1] row 22  real increase in payroll expenses
        volume_growth = at_t(learnables.volume_growth)  # [B, 1, 1] row 23  increase in sales volume, units

        # Market research levels (Table 1c)
        base_selling_price = at_t(learnables.base_selling_price)  # [B, 1, 1] row 40  year-0 price level
        elasticity_b = at_t(learnables.elasticity_b)  # [B, 1, 1] row 41
        elasticity_b0 = at_t(learnables.elasticity_b0)  # [B, 1, 1] row 42

        # --- Accounting / structural conventions ---
        tax_rate_t = at_t(policies.tax_rate)  # [B, 1, 1] row 9 (Table 1a)
        depr_life_t = at_t(policies.depr_life)  # [B, 1, 1] row 7 -> depreciation life in years (straight-line)
        # Optionally keep a book value series if desired:
        fixed_asset = at_t(policies.fixed_asset)  # [B, 1, 1] row 6 (used only if you want to drive NFA externally)

        # --- Working capital timing rails (Table 1b) ---
        # Sales rails
        sales_credit_share = at_t(policies.sales_credit_share)  # [B, 1, 1] row 30  (AR% of sales, collected t+1)
        sales_adv_share = at_t(policies.sales_adv_share)  # [B, 1, 1] row 31  (advances for t+1, received at t)
        sales_same_share = 1 - sales_credit_share - sales_adv_share  # same-year cash share is implied: 1 - credit - advance  (row 35)
            
        # Purchases rails
        purch_credit_share = at_t(policies.purch_credit_share)  # [B, 1, 1] row 32  (AP% of purchases, paid t+1)
        purch_adv_share = at_t(policies.purch_adv_share)  # [B, 1, 1] row 33  (advances to suppliers for t+1, paid at t)

        # --- Inventory policy (Table 1b) ---
        inventory_volume_ratio = at_t(policies.inventory_volume_ratio)  # [B, 1, 1] row 29  (units kept as fraction of units sold)

        # --- Opex policies (Table 1b) ---
        advertisement_sales_ratio = at_t(policies.advertisement_sales_ratio)  # [B, 1, 1] row 28  (ad as % of sales)
        selling_commission_ratio = at_t(policies.selling_commission_ratio)  # [B, 1, 1] row 38  (commissions as % of sales)
        opex_ratio = at_t(policies.opex_ratio)  # [B, 1, 1] row 39  (opex as % of sales)

        # --- Payout / financing policies (Table 1b) ---
        payout_ratio = at_t(policies.payout_ratio)  # [B, 1, 1] row 34
        deficit_debt_share = at_t(policies.deficit_debt_share)  # [B, 1, 1] row 36 (LT deficit funded with debt; rest equity)

        # --- Cash policy (Table 1b) ---
        # Paper gives an initial-year absolute minimum cash (row 37). Our engine mostly uses a ratio.
        min_cash_ratio = at_t(policies.min_cash_ratio)  # [B, 1, 1] row 37
        min_cash_initial_abs = policies.min_cash_initial_abs  # [B, 1, 1] row 37 (initial-year absolute min cash)

        # --- Repurchase rule (Table 1b row 39) ---
        repurchase_share_of_depr = at_t(policies.repurchase_share_of_depr)  # [B, 1, 1] row 39  (share of depreciation used for repurchase)

        # === Implement the structural model step-by-step here ===

        # --- Global initial-year adjustments ---
        
        # (e.g., some variables are set to zero at t=0 instead of following normal logic)
        if is_initial:

            # --- Table 2: Set nominal increases to zero at initial step ---
            nom_inc_sell_t = tf.zeros_like(fixed_asset) # [B,T,1]
            nom_inc_purchase_t = tf.zeros_like(fixed_asset)    # [B,T,1]
            nom_inc_overhead_t = tf.zeros_like(fixed_asset)    # [B,T,1]
            nom_inc_payroll_t = tf.zeros_like(fixed_asset)    # [B,T,1]
            min_cash = min_cash_initial_abs  # [B, 1, 1]

            # --- Table 3: Set initial volume and price ---
            units_t = elasticity_b0 * tf.pow(base_selling_price, elasticity_b)  # [B, 1, 1]
            price_t = base_selling_price  # [B, 1, 1]
            sales_t = units_t * price_t  # [B, 1, 1]

            # --- Table 4: Set initial rates to zero ---
            Rf_t = tf.zeros_like(real_rate)  # [B, 1, 1]
            st_invest_return_t = tf.zeros_like(real_rate)  # [B, 1, 1]
            Kd_t = tf.zeros_like(real_rate)  # [B, 1, 1]

            # ---- TABLE 5: depreciation & capex (one step) ----
            dep_t = tf.zeros_like(fixed_asset)  # [B, 1, 1]
            # capex_repl_t = fixed_asset  # [B, 1, 1]
            # capex_growth_t = tf.zeros_like(fixed_asset)  # [B, 1, 1]
            # capex_t = capex_repl_t + capex_growth_t  # [B, 1, 1]
            nfa_end_t = fixed_asset  # [B, 1, 1]

            # --- Table 6a: Inventory and purchases in units ---
            inv_units_t = 

        else:

        # --- Table 2: Compute nominal increases via Fisher equation ---
            nom_inc_sell_t = fisher_nominal(real_increase_sell_price, inflation)  # [B,T,1]
            nom_inc_purchase_t = fisher_nominal(real_increase_unit_cost, inflation)  # [B,T,1]
            nom_inc_overhead_t = fisher_nominal(real_increase_overhead, inflation)  # [B,T,1]
            nom_inc_payroll_t = fisher_nominal(real_increase_payroll, inflation)  # [B,T,1]

        volume_growth_t = 1 + volume_growth  # [B, 1, 1]
        prev_units = prev.units_sold  # [B, 1, 1]
        prev_price = prev.selling_price  # [B, 1, 1]

        # --- Table 3: Forecast volume, price, sales ---
        # units_t, row 55
        # price_t, row 56
        # sales_t, row 57
        (units_t,
        price_t,
        sales_t) = self.forecast_volume_price_sales_step(
            elasticity_b = elasticity_b,
            elasticity_b0 = elasticity_b0,
            base_selling_price = base_selling_price,
            volume_growth_t = volume_growth_t,
            prev_units = prev_units,
            prev_price = prev_price,
            nom_inc_sell_t = nom_inc_sell_t,
            is_initial = is_initial,
        )
        if is_initial:
            units_t = elasticity_b0 * tf.pow(base_selling_price, elasticity_b)  # [B, 1, 1]
            price_t = base_selling_price  # [B, 1, 1]
        else:
            prev_units = prev.units_sold  # [B, 1, 1]
            prev_price = prev.selling_price  # [B, 1, 1]
            volume_growth_t = 1 + volume_growth  # [B, 1, 1]
            units_t = prev_units * (1 + volume_growth_t)  # [B, 1, 1]
            price_t = prev_price * (1 + nom_inc_sell_t)  # [B, 1, 1]
        sales_t = units_t * price_t  # [B, 1, 1]

        # --- Minimum cash calculation ---
        min_cash = min_cash_ratio * sales_t  # [B, 1, 1]

        # Table 4: Forecast Risk free rate (Rf), cost of debt (Kd), investment returns, minimum cash
        if is_initial:
            Rf_t = tf.zeros_like(real_rate)  # [B, 1, 1]
            st_invest_return_t = tf.zeros_like(real_rate)  # [B, 1, 1]
            Kd_t = tf.zeros_like(real_rate)  # [B, 1, 1]
        else:
            Rf_t = fisher_nominal(real_rate, inflation)  # [B, 1, 1]
            st_invest_return_t = Rf_t + risk_premium_st_invest  # [B, 1, 1]
            Kd_t = Rf_t + risk_premium_debt  # [B, 1, 1]

        # ---- TABLE 5: depreciation & capex (one step) ----
        # row 66-77
        # capex for growth is capped after L years of growth investment
        # capex = capex_replacement + capex_growth
        # capex
        (dep_t,
        capex_repl_t,
        capex_growth_t,
        capex_t,
        nfa_end_t,
        last_depr_next,
        growth_years_next) = capex_depr_step_capped_L(
            prev_nfa=prev.nfa,
            prev_last_depr=prev.last_annual_depr,
            growth_cap_yrs_done=prev.growth_cap_yrs,
            volume_growth_t=volume_growth_t,
            depr_life_t=depr_life_t,
            init_fa = fixed_asset,
            is_initial=is_initial,
        )

        # --- Table 6a: Inventory and purchases in units ---
        # row 82-84
        (final_inv_units_t,
        purchases_units_t) = inventory_purchases_units_step(
            units_sold_t=units_t,
            prev_inv_units=prev.inv_units,
            inventory_volume_ratio=inventory_volume_ratio,
            is_initial=is_initial,
        )

        # --- Table 6b: Inventory valuation using FIFO and COGS in $ ---


