# tf_balance_sheet_model.py
# ----------------------------------------------
# A very small TensorFlow 2.x balance-sheet model
# (implements the tools/assumptions from Vélez-Pareja 2009/2011)
#
# How to run:
#   python tf_balance_sheet_model.py
#
# Requirements:
#   pip install tensorflow pandas numpy
#
# Output: prints a table and writes balance_sheet_results.csv
from __future__ import annotations
import tensorflow as tf
import numpy as np
import pandas as pd

# ---------------------------
# 1) Core building blocks
# ---------------------------
def build_price_path(p0: float, inflation: float, T: int) -> tf.Tensor:
    """Nominal price path using Fisher-like scalar inflation (simple)."""
    t = tf.range(T, dtype=tf.float32)
    return tf.cast(p0, tf.float32) * tf.pow(1.0 + tf.cast(inflation, tf.float32), t)

def build_units_path(q0: float, growth: float, T: int) -> tf.Tensor:
    """Real unit growth path."""
    t = tf.range(T, dtype=tf.float32)
    return tf.cast(q0, tf.float32) * tf.pow(1.0 + tf.cast(growth, tf.float32), t)

def straight_line_depr_matrix(capex: np.ndarray, life: int) -> tf.Tensor:
    """
    Build a (T x T) depreciation matrix (vintages in columns, time in rows).
    Each vintage i depreciates equally for 'life' years starting the period after purchase.
    """
    T = capex.shape[0]
    dep = np.zeros((T, T), dtype=np.float32)
    if life <= 0:
        return tf.constant(dep, dtype=tf.float32)
    for i in range(T):
        amt = capex[i] / life
        # start next period, inclusive, up to i+life-1 (capped at T-1)
        for j in range(i+1, min(T, i+life)+1):
            dep[j-1, i] = amt
    return tf.constant(dep, dtype=tf.float32)

def compute_inventory_units(sales_units: tf.Tensor,
                            inv_months: float,
                            months_in_year: float,
                            inv_units_0: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Inventory-units policy: hold inv_months of *next year's* sales as ending inventory.
    PurchasesUnits_t = Inv_end_t - Inv_beg_t + SalesUnits_t
    """
    next_sales = tf.concat([sales_units[1:], sales_units[-1:]], axis=0)
    inv_target = (inv_months / months_in_year) * next_sales
    inv_units_beg0 = tf.cast(inv_units_0, tf.float32)

    def step(prev, current):
        # Unpack accumulator structure (state, emitted)
        prev_inv_end, _ = prev
        sales_t, inv_target_t = current
        inv_beg = prev_inv_end
        inv_end = inv_target_t
        purchases_units = inv_end - inv_beg + sales_t
        emitted = tf.stack([inv_beg, inv_end, purchases_units], axis=0)
        return (inv_end, emitted)

    # Initializer must match (state, emitted) structure
    init_state = inv_units_beg0
    init_emitted = tf.zeros([3], dtype=tf.float32)
    _, triplets = tf.scan(
        fn=step,
        elems=(sales_units, inv_target),
        initializer=(init_state, init_emitted),
    )
    inv_beg = triplets[:, 0]
    inv_end = triplets[:, 1]
    purchases_units = triplets[:, 2]
    return inv_beg, inv_end, purchases_units

def fifo_current_cost_valuation(inv_end_units: tf.Tensor,
                                inv_beg_units: tf.Tensor,
                                purchases_units: tf.Tensor,
                                unit_cost: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Simple valuation consistent with the paper's basic build:
    COGS_t = Inv_beg_units_t*IC_t + Purchases_t - Inv_end_units_t*IC_t
    """
    purchases_value = purchases_units * unit_cost
    inv_beg_value = inv_beg_units * unit_cost
    inv_end_value = inv_end_units * unit_cost
    cogs = inv_beg_value + purchases_value - inv_end_value
    inventory_value = inv_end_value
    return cogs, inventory_value, purchases_value

def working_capital_sales_blocks(sales_value: tf.Tensor,
                                 s_AR: float, s_ADV: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split sales into cash, AR (credit sales), and advances received (APR)."""
    s_AR = tf.cast(s_AR, tf.float32)    # fraction on AR
    s_ADV = tf.cast(s_ADV, tf.float32)   # fraction on advances
    credit_sales = s_AR * sales_value
    advances_received = s_ADV * sales_value
    cash_sales = (1.0 - s_AR - s_ADV) * sales_value
    return cash_sales, credit_sales, advances_received

def working_capital_purchases_blocks(purchases_value: tf.Tensor,
                                     p_AP: float, p_ADV: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split purchases into cash paid, AP, and advances paid (APP)."""
    p_AP = tf.cast(p_AP, tf.float32)    # fraction on AP
    p_ADV = tf.cast(p_ADV, tf.float32)   # fraction on advances
    purchases_on_credit = p_AP * purchases_value
    advances_paid = p_ADV * purchases_value
    purchases_paid = (1.0 - p_AP - p_ADV) * purchases_value
    return purchases_paid, purchases_on_credit, advances_paid

def debt_schedule(bb0: float,
                  new_issues: np.ndarray,
                  principal: np.ndarray,
                  k_nom: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    End-of-period convention per papers:
      Interest_t = k_nom * BB_t
      EB_t = BB_t - Principal_t + NewIssues_t
    Returns sequences for beginning balance, interest, ending balance.
    """
    k_nom = tf.cast(k_nom, tf.float32)
    bb0 = tf.cast(bb0, tf.float32)

    def step(prev, current):
        bb_prev, _ = prev
        new_t, prin_t = current
        interest_t = k_nom * bb_prev
        eb_t = bb_prev - prin_t + new_t
        # emit prev BB, interest, EB for each t
        return eb_t, tf.stack([bb_prev, interest_t, eb_t], axis=0)

    init_state = bb0
    init_emitted = tf.zeros([3], dtype=tf.float32)
    _, stacked = tf.scan(
        fn=step,
        elems=(tf.cast(new_issues, tf.float32), tf.cast(principal, tf.float32)),
        initializer=(init_state, init_emitted),
    )
    bb = stacked[:, 0]
    interest = stacked[:, 1]
    eb = stacked[:, 2]
    return bb, interest, eb

# ---------------------------
# 2) Model runner
# ---------------------------
def run_model(params: dict) -> pd.DataFrame:
    """
    Vélez-Pareja style deterministic time-series balance-sheet model (Modules 1–5)
    with deficit-driven financing, minimum-cash enforcement, owners’ flows, and
    ST investments (surplus -> invest; shortfall -> redeem).
    Returns a pandas DataFrame with IS/BS lines and a balance check.

    Expected params (defaults provided where sensible):
      T, sales_units_0, unit_price_0, real_growth_q, inflation, unit_input_cost_0,
      as_pct_sales, s_AR, s_ADV, p_AP, p_ADV,
      depr_life, nfa0, inv_units_0, equity0, cash0,
      rf_real, st_debt_spread, lt_debt_spread, stinv_spread,
      debt_share_lt, payout_ratio, tax_rate, min_cash_percent_sales,
      lt_bb0, st_bb0, lt_principal_sched (optional list len T, default zeros)
    """
    # ---------------------------
    # Unpack / defaults
    # ---------------------------
    T = int(params.get("T", 6))

    # Demand & pricing
    sales_units_0 = float(params.get("sales_units_0", 1000.0))
    unit_price_0  = float(params.get("unit_price_0", 10.0))
    real_growth_q  = float(params.get("real_growth_q", 0.05))     # real unit growth q
    inflation      = float(params.get("inflation", 0.03))         # π
    unit_input_cost_0 = float(params.get("unit_input_cost_0", 6.0))

    # Opex and WC splits
    as_pct_sales = float(params.get("as_pct_sales", 0.12))
    s_AR  = float(params.get("s_AR", 0.25))     # sales on AR
    s_ADV = float(params.get("s_ADV", 0.05))    # advances received on sales
    p_AP  = float(params.get("p_AP", 0.40))     # purchases on AP
    p_ADV = float(params.get("p_ADV", 0.05))    # advances paid on purchases

    # Depreciation & stocks
    depr_life = int(params.get("depr_life", 5))
    nfa0      = float(params.get("nfa0", 1000.0))
    inv_units_0 = float(params.get("inv_units_0", 200.0))

    # Opening financing & equity
    equity0 = float(params.get("equity0", 1000.0))
    cash0   = float(params.get("cash0", 100.0))
    st_bb0  = float(params.get("st_bb0", 200.0))
    lt_bb0  = float(params.get("lt_bb0", 500.0))
    lt_principal_sched = np.array(params.get("lt_principal_sched", [0.0]*T), dtype=np.float32)

    # Rates via Fisher + spreads
    rf_real        = float(params.get("rf_real", 0.02))           # real risk-free
    st_debt_spread = float(params.get("st_debt_spread", 0.02))
    lt_debt_spread = float(params.get("lt_debt_spread", 0.03))
    stinv_spread   = float(params.get("stinv_spread", 0.005))

    # Policies
    debt_share_lt = float(params.get("debt_share_lt", 0.6))       # of LT deficit
    payout_ratio  = float(params.get("payout_ratio", 0.30))
    tax_rate      = float(params.get("tax_rate", 0.25))
    min_cash_percent_sales = float(params.get("min_cash_percent_sales", 0.05))

    # ---------------------------
    # Deterministic drivers (paths)
    # ---------------------------
    price_t = build_price_path(unit_price_0, inflation, T)                   # (T,)
    units_t = build_units_path(sales_units_0, real_growth_q, T)              # (T,)
    sales_value_t = price_t * units_t

    # Unit input cost path (current-cost)
    unit_cost_t = build_price_path(unit_input_cost_0, inflation, T)

    # Operating expense
    as_expense_t = as_pct_sales * sales_value_t

    # ---------------------------
    # Per-period recursive step (Modules 1–5)
    # ---------------------------
    def step(state, t):
        (cash_prev, st_bb_prev, lt_bb_prev, stinv_bb_prev,
         nfa_prev, inv_units_prev, equity_prev,
         ar_prev, ap_prev, app_prev, apr_prev, inv_value_prev,
         ni_prev) = state

        # Rates for t (Fisher)
        rf_nom_t = (1.0 + rf_real) * (1.0 + inflation) - 1.0
        kd_st_t  = rf_nom_t + st_debt_spread
        kd_lt_t  = rf_nom_t + lt_debt_spread
        r_stinv_t = rf_nom_t + stinv_spread

        # Sales / Purchases / Inventory valuation
        sales_t = sales_value_t[t]
        # Inventory mechanics & COGS (uses your helper to get purchases and ending inventory at current cost)
        # We keep end units proportional to next period demand to avoid depletion; simplest policy: keep days == 0 (no change)
        # If you want a richer inventory policy, substitute here.
        inv_end_units_t = inv_units_prev  # keep steady units by default
        purchases_units_t = tf.maximum(units_t[t] + inv_end_units_t - inv_units_prev, 0.0)

        cogs_ti, inv_value_t, purchases_value_t = cogs_from_inventory_blocks(
            inv_beg_units=tf.cast(inv_units_prev, tf.float32),
            purchases_units=tf.cast(purchases_units_t, tf.float32),
            inv_end_units=tf.cast(inv_end_units_t, tf.float32),
            unit_cost=tf.cast(unit_cost_t[t], tf.float32),
        )

        # Working capital splits (stocks at end of t)
        cash_sales_t, credit_sales_t, advances_received_t = working_capital_sales_blocks(
            sales_t, s_AR, s_ADV
        )
        purchases_paid_t, purchases_on_credit_t, advances_paid_t = working_capital_purchases_blocks(
            purchases_value_t, p_AP, p_ADV
        )

        # ΔWC components (asset increases are cash OUT; liability increases are cash IN)
        ar_t  = credit_sales_t
        ap_t  = purchases_on_credit_t
        app_t = advances_paid_t      # asset
        apr_t = advances_received_t  # liability

        dAR  = ar_t  - ar_prev
        dAP  = ap_t  - ap_prev
        dAPP = app_t - app_prev
        dAPR = apr_t - apr_prev
        dInv = inv_value_t - inv_value_prev
        delta_WC_cash = (dAR + dAPP + dInv) - (dAP + dAPR)

        # Depreciation (straight-line on prior NFA)
        dep_t = nfa_prev / depr_life

        # EBIT (accrual)
        ebit_t = sales_t - cogs_ti - as_expense_t[t] - dep_t

        # Finance items on beginning balances
        st_interest_t = kd_st_t * st_bb_prev
        lt_interest_t = kd_lt_t * lt_bb_prev
        stinv_return_t = r_stinv_t * stinv_bb_prev   # finance income

        # EBT and Tax (paper taxes in the same period)
        ebt_t = ebit_t - st_interest_t - lt_interest_t + stinv_return_t
        tax_t = tf.maximum(0.0, tax_rate * ebt_t)
        ni_t  = ebt_t - tax_t

        # Dividends paid this period (lagged NI)
        dividends_t = payout_ratio * tf.maximum(0.0, ni_prev)

        # ---------------------------
        # Module 1: Operating NCB (after tax, before capex & financing)
        # Use accrual-to-cash via ΔWC and add back Depreciation (non-cash)
        # ---------------------------
        ncb_oper_t = (ebit_t - tax_t) + dep_t - delta_WC_cash

        # ---------------------------
        # Module 2: Investing NCB (Capex policy = keep-up + growth top-up)
        # ---------------------------
        capex_keep_t  = dep_t
        capex_growth_t = tf.maximum(real_growth_q, 0.0) * nfa_prev
        capex_t = capex_keep_t + capex_growth_t
        ncb_after_capex_t = ncb_oper_t - capex_t

        # ---------------------------
        # Module 3: External financing (LT deficit funded by policy split)
        # ---------------------------
        lt_deficit_t = tf.maximum(0.0, -ncb_after_capex_t)
        lt_new_t = debt_share_lt * lt_deficit_t
        ie_t     = (1.0 - debt_share_lt) * lt_deficit_t   # invested equity (owners)

        # Principal on LT (optional schedule)
        lt_prin_t = tf.cast(lt_principal_sched[t], tf.float32)
        lt_eb_t   = lt_bb_prev + lt_new_t - lt_prin_t
        # (interest already computed off lt_bb_prev)

        # Cash effect so far
        cash_before_st_layer = cash_prev + ncb_after_capex_t + lt_new_t + ie_t - lt_prin_t

        # ---------------------------
        # Module 4: Owners’ cash outflows (dividends in cash)
        # ---------------------------
        cash_after_owners = cash_before_st_layer - dividends_t

        # ---------------------------
        # Module 5: Minimum cash + ST investments + ST loans (no plugs)
        # (1) Redeem ST investments if cash < target
        # (2) If still short, borrow ST to reach target
        # (3) If surplus, invest in ST securities
        # ---------------------------
        target_cash_t = min_cash_percent_sales * sales_t

        # First, redemption to meet target
        shortfall = tf.maximum(0.0, target_cash_t - cash_after_owners)
        redeem_t  = tf.minimum(stinv_bb_prev, shortfall)
        cash_after_redeem = cash_after_owners + redeem_t

        # Then, ST borrowing if still short
        remaining_shortfall = tf.maximum(0.0, target_cash_t - cash_after_redeem)
        st_new_t = remaining_shortfall

        # If surplus, invest into ST securities
        cash_after_st_borrow = cash_after_redeem + st_new_t
        surplus = tf.maximum(0.0, cash_after_st_borrow - target_cash_t)
        st_invest_t = surplus

        # End-of-period balances for ST layers
        st_eb_t    = st_bb_prev + st_new_t  # (no scheduled principal; revolving)
        stinv_eb_t = stinv_bb_prev + st_invest_t - redeem_t

        # End cash = target policy
        cash_end_t = target_cash_t

        # Update balance-sheet stocks
        nfa_t = nfa_prev + capex_t - dep_t

        equity_t = equity_prev + ni_t - dividends_t + ie_t

        # Emit period data (collect in a tuple)
        emitted = (
            sales_t, cogs_ti, as_expense_t[t], dep_t, ebit_t,
            st_interest_t + lt_interest_t, tax_t, ni_t,
            dividends_t, ie_t,
            capex_t, stinv_return_t,
            ncb_oper_t, ncb_after_capex_t,
            cash_end_t, st_invest_t, redeem_t, st_new_t,
            st_eb_t, lt_eb_t, stinv_eb_t,
            ar_t, ap_t, app_t, apr_t, inv_value_t,
            equity_t,
        )

        # Next state
        next_state = (
            cash_end_t, st_eb_t, lt_eb_t, stinv_eb_t,
            nfa_t, inv_end_units_t, equity_t,
            ar_t, ap_t, app_t, apr_t, inv_value_t,
            ni_t
        )
        return next_state, emitted

    # ---------------------------
    # Initialize state and scan across periods
    # ---------------------------
    init_state = (
        tf.cast(cash0, tf.float32),
        tf.cast(st_bb0, tf.float32),
        tf.cast(lt_bb0, tf.float32),
        tf.cast(0.0, tf.float32),                 # ST investments opening balance
        tf.cast(nfa0, tf.float32),
        tf.cast(inv_units_0, tf.float32),
        tf.cast(equity0, tf.float32),
        tf.cast(0.0, tf.float32),  # AR
        tf.cast(0.0, tf.float32),  # AP
        tf.cast(0.0, tf.float32),  # APP
        tf.cast(0.0, tf.float32),  # APR
        tf.cast(inv_units_0 * unit_input_cost_0, tf.float32),  # Inv value
        tf.cast(0.0, tf.float32),  # NI_{t-1}
    )

    _, stacked = tf.scan(step, elems=tf.range(T), initializer=init_state)

    (sales, cogs, as_exp, dep, ebit,
     interest, tax, ni,
     dividends, ie,
     capex, stinv_ret,
     ncb_oper, ncb_after_capex,
     cash_end, st_invest, st_redeem, st_new,
     st_debt_eb, lt_debt_eb, stinv_eb,
     ar, ap, app, apr, inv_value,
     equity) = [stacked[:, i] for i in range(stacked.shape[1])]

    # ---------------------------
    # Assemble Balance Sheet & Check
    # ---------------------------
    assets_cash_t   = cash_end
    assets_stinv_t  = stinv_eb
    assets_ar_t     = ar
    assets_app_t    = app
    assets_inv_t    = inv_value
    # Net fixed assets tracked in the state but not emitted—rebuild quickly via recursion:
    # (NFA_t = NFA_{t-1} + Capex_t - Dep_t)
    def nfa_reduce(prev, x):
        cap, d = x
        return prev + cap - d
    nfa_series = tf.scan(nfa_reduce, (capex, dep), initializer=tf.cast(nfa0, tf.float32))

    assets_total = assets_cash_t + assets_stinv_t + assets_ar_t + assets_app_t + assets_inv_t + nfa_series

    liab_ap_t   = ap
    liab_apr_t  = apr
    liab_st_t   = st_debt_eb
    liab_lt_t   = lt_debt_eb
    liab_total  = liab_ap_t + liab_apr_t + liab_st_t + liab_lt_t

    equity_t = equity

    check_t = assets_total - (liab_total + equity_t)

    # ---------------------------
    # Pandas table
    # ---------------------------
    to_np = lambda x: x.numpy().astype(float)
    df = pd.DataFrame({
        "Year": np.arange(T),
        "Sales": np.round(to_np(sales), 2),
        "COGS": np.round(to_np(cogs), 2),
        "A&S": np.round(to_np(as_exp), 2),
        "Dep": np.round(to_np(dep), 2),
        "EBIT": np.round(to_np(ebit), 2),
        "Interest": np.round(to_np(interest), 2),
        "Tax": np.round(to_np(tax), 2),
        "NI": np.round(to_np(ni), 2),
        "Dividends (t, paid)": np.round(to_np(dividends), 2),
        "IE (Owners)": np.round(to_np(ie), 2),
        "ST-inv return": np.round(to_np(stinv_ret), 2),
        "Capex": np.round(to_np(capex), 2),
        "NCB Oper": np.round(to_np(ncb_oper), 2),
        "NCB After Capex": np.round(to_np(ncb_after_capex), 2),
        "Cash (End)": np.round(to_np(cash_end), 2),
        "ST Invest (new)": np.round(to_np(st_invest), 2),
        "ST Invest (redeem)": np.round(to_np(st_redeem), 2),
        "ST Debt (new)": np.round(to_np(st_new), 2),
        "ST Debt (End)": np.round(to_np(st_debt_eb), 2),
        "LT Debt (End)": np.round(to_np(lt_debt_eb), 2),
        "ST Investments (End)": np.round(to_np(stinv_eb), 2),
        "AR": np.round(to_np(ar), 2),
        "AP": np.round(to_np(ap), 2),
        "APP (advances paid)": np.round(to_np(app), 2),
        "APR (advances recvd)": np.round(to_np(apr), 2),
        "Inventory (value)": np.round(to_np(inv_value), 2),
        "NFA": np.round(to_np(nfa_series), 2),
        "Equity": np.round(to_np(equity_t), 2),
        "BS Check": np.round(to_np(check_t), 6),
    })
    return df
