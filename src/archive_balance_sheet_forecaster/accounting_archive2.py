from __future__ import annotations
from typing import Tuple, Optional, List
import tensorflow as tf

from balance_sheet_forecaster.types import (
    Policies, Drivers, PrevState, Statements
)

# === Utility ===

def fisher_nominal(real_rate: tf.Tensor, inflation: tf.Tensor) -> tf.Tensor:
    # (1+r_nom) = (1+r_real)*(1+pi) - 1
    return (1.0 + tf.nn.relu(real_rate)) * (1.0 + inflation) - 1.0

def bcast_like(x: tf.Tensor, ref: tf.Tensor) -> tf.Tensor:
    # expand [B,1,1] to [B,T,1] by tiling to ref shape
    tiles = tf.concat([tf.ones_like(ref[:, :1, :], dtype=tf.int32) * tf.shape(ref)[0:1],
                       tf.ones_like(ref[:, :1, :], dtype=tf.int32) * tf.shape(ref)[1:2],
                       tf.ones_like(ref[:, :1, :], dtype=tf.int32) * tf.shape(ref)[2:3]], axis=0)
    # Simpler and stable: just tile along time
    return tf.tile(x, [1, tf.shape(ref)[1], 1])

def clamp01(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)

def zero_like(ref: tf.Tensor) -> tf.Tensor:
    return tf.zeros_like(ref)

def shift_fwd(x: tf.Tensor) -> tf.Tensor:
    # shift forward in time by 1: [B,T,1], prepend zeros, drop last
    pad = tf.zeros_like(x[:, :1, :])
    return tf.concat([pad, x[:, :-1, :]], axis=1)

def shift_back(x: tf.Tensor) -> tf.Tensor:
    # shift backward: append zeros at the end
    pad = tf.zeros_like(x[:, :1, :])
    return tf.concat([x[:, 1:, :], pad], axis=1)

# === FIFO Inventory helper ===

class FIFOInventory:
    """
    Tensor-friendly FIFO valuation with a single layer state: we store only total units and total value.
    COGS is valued by drawing units at average *of existing FIFO stack*; this is a pragmatic compromise
    (if you need strict multi-layer queues per purchase, we can extend—this version stays differentiable and fast).
    """

    @staticmethod
    def draw_cogs(beg_units: tf.Tensor, beg_value: tf.Tensor, sell_units: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        beg_units/value: [B,T,1] beginning-of-period inventory
        sell_units: [B,T,1] units sold in period
        Returns: (cogs_value, end_units, end_value)
        """
        # Avoid negative: can’t sell more than you have + purchases (purchases handled outside; draw on beg first)
        draw_units = tf.minimum(tf.nn.relu(beg_units), tf.nn.relu(sell_units))

        avg_cost = tf.where(beg_units > 0.0, beg_value / tf.maximum(beg_units, 1e-8), tf.zeros_like(beg_units))
        cogs_from_beg = draw_units * avg_cost

        end_units = beg_units - draw_units
        end_value = tf.nn.relu(beg_value - cogs_from_beg)

        return cogs_from_beg, end_units, end_value

# === Structural (deterministic) layer faithful to Vélez-Pareja ===

class StructuralLayer(tf.keras.layers.Layer):
    def __init__(self, name: str = "structural_layer"):
        super().__init__(name=name)

    def call(
        self,
        drivers: Drivers,       # [B,T,1] fields
        policies: Policies,     # [B,T,1] fields (or None for optional)
        prev: PrevState,        # [B,1,1] fields
        training: bool = False
    ) -> Statements:

        # Unpack drivers
        price, volume, unit_cost = drivers.price, drivers.volume, drivers.unit_cost
        units_sold = tf.nn.relu(volume)
        sales = tf.nn.relu(price) * units_sold  # [B,T,1]

        # Opex
        if drivers.opex is not None:
            opex = tf.nn.relu(drivers.opex)
        else:
            opex_ratio = policies.opex_ratio if policies.opex_ratio is not None else (tf.ones_like(sales) * 0.2)
            opex = tf.nn.relu(opex_ratio) * sales

        # Rates
        nominal = fisher_nominal(policies.real_rate, policies.inflation)
        lt_rate = policies.lt_rate if policies.lt_rate is not None else (nominal + 0.015)
        st_borrow = policies.st_borrow_rate if policies.st_borrow_rate is not None else nominal
        st_invest = policies.st_invest_rate if policies.st_invest_rate is not None else nominal

        # === INVENTORY & COGS (FIFO-like) ===
        # Target ending inventory in units: simple policy as a fraction of next period sales units (you can swap policy)
        # For end-of-period convention, we’ll set a mild buffer (e.g., keep 0.0 by default). Replace with your explicit policy if desired.
        inventory_volume_ratio = policies.inventory_volume_ratio if policies.inventory_volume_ratio is not None else tf.zeros_like(units_sold)
        inventory_target_units = inventory_volume_ratio * units_sold  # [B,T,1], target ending inventory in units

        # Purchases in units = units sold + target - beginning (per period)
        beg_units_T = bcast_like(prev.inventory_units, units_sold)
        beg_value_T = bcast_like(prev.inventory_value, units_sold)

        # First draw COGS from beginning inventory at its average cost
        cogs_from_beg, after_units, after_value = FIFOInventory.draw_cogs(beg_units_T, beg_value_T, units_sold)

        # Now compute purchases to reach target ending units:
        purchases_units = tf.nn.relu(units_sold + inventory_target_units - after_units)
        purchases_value = purchases_units * tf.nn.relu(unit_cost)

        # After adding purchases (at current unit cost), final ending inventory value
        end_units = after_units + purchases_units
        end_value = after_value + purchases_value

        # Total COGS value is: BegInv + Purchases − EndInv (by identity)
        cogs = tf.nn.relu(beg_value_T + purchases_value - end_value)

        # === SALES & PURCHASES TIMING RAILS ===
        # Sales shares
        s_adv = clamp01(policies.sales_adv_share)   # advances received
        s_credit = clamp01(policies.sales_credit_share) # on credit (AR)
        s_cash = clamp01(1.0 - s_adv - s_credit)    # cash sales

        # Purchases shares (on purchases_value, not COGS)
        p_adv = clamp01(policies.purch_adv_share)   # supplier advances paid
        p_credit = clamp01(policies.purch_credit_share) # on credit (AP)
        p_cash = clamp01(1.0 - p_adv - p_credit)    # cash paid

        # Rails for period t:
        # Sales: advances are assumed received at t-1 for sales of t (so cash + advances_received stock move at t-1)
        # In period t cash-in is: cash sales now + collection of prior AR
        sales_cash_same = s_cash * sales
        sales_credit_now = s_credit * sales            # becomes AR at t, collected at t+1
        sales_adv_for_t = s_adv * sales                # were received at t-1 and sit in advances_received; reverse now

        # Purchases: advances are paid at t for next period’s purchases
        purch_cash_same = p_cash * purchases_value
        purch_credit_now = p_credit * purchases_value  # becomes AP at t, paid at t+1
        purch_adv_for_tp1 = p_adv * purchases_value    # paid now, recognized as advance paid

        # Stocks roll (AR/AP/Advances) with one-period lags/leads
        # Ending AR_t = sales_credit_now
        ar = sales_credit_now
        # Collections (cash in) from AR_{t-1} happen this period:
        ar_collect = shift_fwd(sales_credit_now)  # AR built at t-1 collected at t

        # Ending AP_t = purch_credit_now
        ap = purch_credit_now
        # Payments (cash out) for AP_{t-1}:
        ap_pay = shift_fwd(purch_credit_now)

        # Customer advances (liability): received at t-1 for sales of t
        # We reverse (recognize revenue) at t; ending advances_received_t is share for sales of t+1
        advances_received = shift_back(sales_adv_for_t)  # liability built at t for future revenue at t+1

        # Supplier advances (asset): paid at t for purchases delivered at t+1
        advances_paid = purch_adv_for_tp1

        # === EBIT, TAX, NI ===
        ebit = sales - cogs - opex

        # Interest on prior balances (no circularity)
        prev_cash_T = bcast_like(prev.cash, sales)
        prev_stinv_T = bcast_like(prev.st_investments, sales)
        prev_stdebt_T = bcast_like(prev.st_debt, sales)
        prev_ltdebt_T = bcast_like(prev.lt_debt, sales)

        interest_st_debt = prev_stdebt_T * st_borrow
        interest_lt_debt = prev_ltdebt_T * lt_rate
        interest_st_inv = - prev_stinv_T * st_invest  # income reduces interest expense

        interest = interest_st_debt + interest_lt_debt + interest_st_inv

        taxable_income = ebit - interest
        tax = tf.nn.relu(taxable_income * policies.tax_rate)
        net_income = taxable_income - tax

        # === CASH FLOWS (rails) ===
        # Cash inflows this period:
        cash_in = sales_cash_same + ar_collect + advances_received  # + other inflows as needed

        # Reverse customer advances for current sales: these were received in t-1, so no cash this year; just release liability.
        # Cash outflows this period:
        cash_out_oper = purch_cash_same + ap_pay + opex + tax + advances_paid

        # CAPEX / Depreciation (simple straight line via policy on replacement if provided)
        # For faithful structure: let repurchase be a policy tie to depreciation
        depr_rate = policies.depr_rate if policies.depr_rate is not None else tf.ones_like(sales) * 0.05
        if policies.repurchase_share_of_depr is not None:
            # simple depreciation proxy: a fraction of prior NFA
            prev_nfa_T = bcast_like(prev.nfa, sales)
            depreciation = prev_nfa_T * depr_rate
            capex = tf.nn.relu(policies.repurchase_share_of_depr) * depreciation
        else:
            prev_nfa_T = bcast_like(prev.nfa, sales)
            depreciation = prev_nfa_T * depr_rate
            capex = tf.zeros_like(sales)
        # NFA roll-forward
        nfa = tf.nn.relu(prev_nfa_T - depreciation + capex)

        # Dividends paid: payout_ratio_{t-1} * NI_{t-1}
        payout_ratio = policies.payout_ratio
        div_declared_tminus1 = shift_fwd(payout_ratio * net_income)   # amount declared last period and paid now
        dividends_paid = div_declared_tminus1   # dividends declared last period, paid now
        dividends_payable = payout_ratio * net_income  # declared now, payable next period

        # Pre-financing cash budget
        ncb = cash_in - cash_out_oper - capex - dividends_paid

        # Cash prior to financing activities
        pre_cash = prev_cash_T + ncb

        # === FINANCING (no cash plug; LT deficit allocation) ===
        # Surplus → ST investments; Deficit → split into LT debt (fixed share) and equity issuance
        surplus = tf.nn.relu(pre_cash)
        deficit = tf.nn.relu(-pre_cash)

        # Additions:
        lt_deficit_share = clamp01(policies.lt_deficit_debt_share)
        add_lt_debt = lt_deficit_share * deficit
        add_equity = (1.0 - lt_deficit_share) * deficit

        # ST investments increase only if surplus; if deficit, they are not increased (could be reduced if you want).
        st_investments = tf.where(deficit > 0.0, tf.zeros_like(surplus), surplus)

        # End-of-period cash equals zero if we parked surplus or funded deficit completely (end-of-period convention).
        # If you prefer to carry some terminal cash, make that a policy. Here we keep pre_cash resolved:
        cash = tf.where(pre_cash > 0.0, tf.zeros_like(pre_cash), tf.zeros_like(pre_cash))

        # Debt roll-forward
        st_debt = prev_stdebt_T  # unchanged (all deficit funded long-term per paper’s LT-deficit rule)
        lt_debt = prev_ltdebt_T + add_lt_debt

        # Equity roll-forward
        equity = bcast_like(prev.equity, sales) + net_income - dividends_paid + add_equity

        # ST investments final (already set)

        # AR/AP/Advances already computed as end-of-period balances
        # Inventory end balances in units/value computed earlier
        stm = Statements(
            sales=sales,
            cogs=cogs,
            opex=opex,
            ebit=ebit,
            interest=interest,
            tax=tax,
            net_income=net_income,
            cash=cash,
            st_investments=st_investments,
            ar=ar,
            ap=ap,
            advances_received=advances_received,
            advances_paid=advances_paid,
            inventory_units=end_units,
            inventory_value=end_value,
            st_debt=st_debt,
            lt_debt=lt_debt,
            nfa=nfa,
            equity=equity,
            dividends_payable=dividends_payable,
            ncb=ncb,
        )

        # Identity check (soft): A == L+E
        if training:
            # Keep it soft to avoid false fails with simplified FIFO; we’ll use a guardrail loss in training.
            tf.debugging.assert_all_finite(stm.assets, "Assets had non-finite values")
            tf.debugging.assert_all_finite(stm.liab_plus_equity, "L+E had non-finite values")
            diff = tf.abs(stm.assets - stm.liab_plus_equity)
            tf.debugging.assert_less_equal(diff, 1e-2, message="Balance sheet identity A = L + E violated beyond tolerance")

        return stm
