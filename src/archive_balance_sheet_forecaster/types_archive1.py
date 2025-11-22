from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import tensorflow as tf


Tensor = tf.Tensor  # [B, T, 1] unless specified; PrevState fields are [B, 1]

@dataclass
class Learnables:
    # Macro / rates (from Pareja et al. 2009, Table 1a)
    initial_purchase_price: Tensor  # [B,1,1] row 11 initial purchase/unit cost at t=0
    estimate_overhead_expense: Tensor  # [B,1,1] row 12 initial overhead expense at t=0
    admin_sales_payroll: Tensor      # [B,1,1] row 13 initial admin & sales payroll at t=0

    inflation: Tensor       # [B,T,1]   row 18 general inflation (for unit-cost indexation)
    real_rate: Tensor       # [B,T,1]   row 24 real ST rate, Fisher gives nominal
    risk_premium_debt: Tensor         # [B,T,1] row 25  adds to nominal for LT/ST borrow
    risk_premium_st_invest: Tensor    # [B,T,1] row 26  adds to nominal for ST investments

    # Real (above-inflation) drifts (Table 1a)
    real_increase_sell_price: Tensor  # [B,T,1] row 19  real increase in selling price
    real_increase_unit_cost: Tensor   # [B,T,1] row 20  real increase in purchase/unit cost
    real_increase_overhead: Tensor    # [B,T,1] row 21  real increase in overhead expenses
    real_increase_payroll: Tensor     # [B,T,1] row 22  real increase in payroll expenses
    volume_growth: Tensor             # [B,T,1] row 23  increase in sales volume, units

    # Market research levels (Table 1c)
    base_selling_price: Tensor        # [B,T,1] row 40  year-0 price level
    elasticity_b: Optional[Tensor] = None   # [B,T,1] row 41
    elasticity_b0: Optional[Tensor] = None  # [B,T,1] row 42


@dataclass
class Policies:
    # --- Accounting / structural conventions ---
    # Optionally keep a book value series if desired:
    fixed_asset: Optional[Tensor] = None   # [B,T,1] row 6 (used only if you want to drive NFA externally)
    
    tax_rate: Tensor                  # [B,T,1] row 9 (Table 1a)
    depr_life: Tensor                  # [B,T,1] row 7 -> depreciation life L in years (straight-line)

    lt_loan_term_yrs: Optional[Tensor] = None  # [B,T,1] row 15 (if modeling explicit amortization schedules)
    st_loan_term_yrs: Optional[Tensor] = None  # [B,T,1] row 16 (if modeling explicit amortization schedules)

    # --- Working capital timing rails (Table 1b) ---
    # Sales rails
    sales_credit_share: Tensor        # [B,T,1] row 30  (AR% of sales, collected t+1)
    sales_adv_share: Tensor           # [B,T,1] row 31  (advances for t+1, received at t)
    # same-year cash share is implied: 1 - credit - advance  (row 35)
    # Purchases rails
    purch_credit_share: Tensor        # [B,T,1] row 32  (AP% of purchases, paid t+1)
    purch_adv_share: Tensor           # [B,T,1] row 33  (advances to suppliers for t+1, paid at t)

    # --- Inventory policy (Table 1b) ---
    inventory_volume_ratio: Tensor    # [B,T,1] row 29  (units kept as fraction of units sold)

    # --- Opex policies (Table 1b) ---
    advertisement_sales_ratio: Tensor # [B,T,1] row 28  (ad as % of sales)
    selling_commission_ratio: Tensor  # [B,T,1] row 38  (commissions as % of sales)
    opex_ratio: Optional[Tensor] = None  # keep as you have (fallback if not modeling overhead/payroll separately)

    # --- Payout / financing policies (Table 1b) ---
    payout_ratio: Tensor              # [B,T,1] row 34
    deficit_debt_share: Tensor        # [B,T,1] row 36 (LT deficit funded with debt; rest equity)

    # --- Cash policy (Table 1b) ---
    # Paper gives an initial-year absolute minimum cash (row 37). Our engine mostly uses a ratio.
    min_cash_ratio: Tensor            # [B,T,1]     # our engine uses this primarily
    min_cash_initial_abs: Optional[Tensor] = None  # [B,1,1] for year 0 absolute requirement (optional), paper setup.

    # --- Rates used by the structural layer (can be derived from Drivers) ---
    # If we pass Drivers, you can compute these as:
    #   nominal = fisher(real_rate, inflation)
    #   lt_rate = nominal + risk_premium_debt, etc.
    lt_rate: Optional[Tensor] = None
    st_borrow_rate: Optional[Tensor] = None
    st_invest_rate: Optional[Tensor] = None

    # --- Repurchase rule (Table 1b row 39) ---
    repurchase_share_of_depr: Optional[Tensor] = None  # [B,T,1] row 39

    # # Convenience: subscriptable by batch/time
    # def slice_t(self, t: int) -> "Policies":
    #     def sel(v):
    #         if v is None: return None
    #         return v[:, t:t+1, :]
    #     return Policies(
    #         inflation=sel(self.inflation), real_rate=sel(self.real_rate), tax_rate=sel(self.tax_rate),
    #         payout_ratio=sel(self.payout_ratio),
    #         advertisement_sales_ratio=sel(self.advertisement_sales_ratio), inventory_volume_ratio=sel(self.inventory_volume_ratio),
    #         sales_adv_share=sel(self.sales_adv_share), sales_credit_share=sel(self.sales_credit_share),
    #         purch_adv_share=sel(self.purch_adv_share), purch_credit_share=sel(self.purch_credit_share),
    #         deficit_debt_share=sel(self.deficit_debt_share), min_cash_ratio=sel(self.min_cash_ratio),
    #         lt_rate=sel(self.lt_rate) if self.lt_rate is not None else None,
    #         st_borrow_rate=sel(self.st_borrow_rate) if self.st_borrow_rate is not None else None,
    #         st_invest_rate=sel(self.st_invest_rate) if self.st_invest_rate is not None else None,
    #         opex_ratio=sel(self.opex_ratio) if self.opex_ratio is not None else None,
    #         repurchase_share_of_depr=sel(self.repurchase_share_of_depr) if self.repurchase_share_of_depr is not None else None,
    #     )

    # # Optional numpy-like indexing (batch/time only, not features)
    # def __getitem__(self, idx) -> "Policies":
    #     def ix(v):
    #         if v is None: return None
    #         return v[idx]
    #     return Policies(
    #         inflation=ix(self.inflation), real_rate=ix(self.real_rate), tax_rate=ix(self.tax_rate),
    #         payout_ratio=ix(self.payout_ratio), advertisement_sales_ratio=ix(self.advertisement_sales_ratio),
    #         inventory_volume_ratio=ix(self.inventory_volume_ratio),
    #         sales_adv_share=ix(self.sales_adv_share), sales_credit_share=ix(self.sales_credit_share),
    #         purch_adv_share=ix(self.purch_adv_share), purch_credit_share=ix(self.purch_credit_share),
    #         deficit_debt_share=ix(self.deficit_debt_share), min_cash_ratio=ix(self.min_cash_ratio),
    #         lt_rate=ix(self.lt_rate) if self.lt_rate is not None else None,
    #         st_borrow_rate=ix(self.st_borrow_rate) if self.st_borrow_rate is not None else None,
    #         st_invest_rate=ix(self.st_invest_rate) if self.st_invest_rate is not None else None,
    #         opex_ratio=ix(self.opex_ratio) if self.opex_ratio is not None else None,
    #         repurchase_share_of_depr=ix(self.repurchase_share_of_depr) if self.repurchase_share_of_depr is not None else None,
    #     )


@dataclass
class PrevState:
    t: Optional[Tensor] = None  # time index (optional, for reference)
    growth_cap_yrs: Optional[Tensor] = None  # [B,1,1] int32 count of years of growth capex so far (optional)

    # prior-period realized units and price (to build t via one-step growth/increase)
    units_sold: Tensor = None                    # [B,1,1]
    selling_price: Tensor = None                 # [B,1,1]
    capex_cohorts: Optional[Tensor] = None        
    # [B,L,1] (optional) capex (investment in new assets) cohorts for NFA rollforward
    # L = number of depreciation cohorts (e.g., 4 for 4-year straight-line)

    # === Opening cash & equivalents (beginning-of-period for t) ===
    cash: Tensor = None                               # [B,1,1] operating cash (BOP)
    st_investments: Tensor = None                     # [B,1,1] short-term investable cash (BOP)

    # === Working-capital rails (opening stocks at t) ===
    accounts_receivable: Tensor = None                # [B,1,1] AR from sales at t-1, collected at t
    advances_from_customers: Tensor = None            # [B,1,1] liability: advances received at t-1 for sales at t
    accounts_payable: Tensor = None                   # [B,1,1] AP from purchases at t-1, paid at t
    advances_to_suppliers: Tensor = None              # [B,1,1] asset: advances paid at t-1 for purchases at t

    # === Inventory opening position for t (valuation inputs for FIFO/COGS) ===
    inv_units: Tensor = None                          # [B,1,1] beginning inventory units, row 10 initial inventory (units) at t=0
    inv_unit_cost: Tensor = None                      # [B,1,1] average/FIFO layer unit cost for opening stock, row11 initial purchase/unit cost at t=0
    # If you model explicit FIFO layers elsewhere, keep inv_unit_cost as the
    # effective carry cost of the oldest layer available at t (or maintain your
    # layer queue outside PrevState). Avoid storing both units*and*value to
    # prevent double-booking—derive value = inv_units * inv_unit_cost when needed.

    # === Long-lived assets (opening book at t) ===
    nfa: Tensor = None                                # [B,1,1] Net fixed assets (book), BOP for t, set 0 at t=0. row 6
    last_annual_depr: Tensor                          # [B,1,1] (set 0 at t=0)
    # (If you track gross & accumulated depreciation separately, keep NFA as the net opening amount.)

    # === Debt balances (interest computed on beginning-of-period balances) ===
    st_loan_principal: Tensor = None                  # [B,1,1] short-term loan balance at BOP t
    lt_debt_principal: Tensor = None                  # [B,1,1] long-term debt principal at BOP t

    # === Equity & lagged flows affecting current cash/owners’ module ===
    equity_book: Tensor = None                        # [B,1,1] total equity at BOP t (book)
    dividends_payable: Tensor = None                  # [B,1,1] declared at t-1, paid at t
    last_net_income: Tensor = None                    # [B,1,1] NI_{t-1} to compute dividends declared at t

    # (Optional) taxes payable if you later change timing (paper pays same year)
    taxes_payable: Optional[Tensor] = None            # [B,1,1]


@dataclass
class Statements:
    # Operating
    sales: Tensor
    cogs: Tensor
    opex: Tensor
    ebit: Tensor
    interest: Tensor
    tax: Tensor
    net_income: Tensor

    # Cash & working capital stocks
    cash: Tensor
    st_investments: Tensor
    ar: Tensor
    ap: Tensor
    advances_received: Tensor
    advances_paid: Tensor
    inventory_units: Tensor
    inventory_value: Tensor

    # Debt & fixed assets & equity
    st_debt: Tensor
    lt_debt: Tensor
    nfa: Tensor
    equity: Tensor
    dividends_payable: Tensor

    # Diagnostics
    ncb: Tensor  # net cash budget (pre financing)

    @property
    def assets(self) -> Tensor:
        # Current assets: cash + st investments + AR + advances paid + inventory value
        ca = self.cash + self.st_investments + self.ar + self.advances_paid + self.inventory_value
        # Non-current assets: NFA
        return ca + self.nfa

    @property
    def liab_plus_equity(self) -> Tensor:
        # Current liabilities: AP + advances received + dividends payable
        cl = self.ap + self.advances_received + self.dividends_payable
        # Non-current liabilities: LT debt; Equity: equity
        return cl + self.st_debt + self.lt_debt + self.equity
