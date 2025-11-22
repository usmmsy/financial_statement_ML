from dataclasses import dataclass
from typing import Optional
import tensorflow as tf

# Tensor shapes convention:
# - Time series: [B, T, 1]
# - Previous state (single slice): [B, 1]

@dataclass
class PoliciesWMT:
    # --- Macro and rate paths ---
    inflation: tf.Tensor                # [B, T, 1]
    real_st_rate: tf.Tensor             # [B, T, 1]
    real_lt_rate: tf.Tensor             # [B, T, 1]

    # --- Tax and payout policies ---
    tax_rate: tf.Tensor                 # [B, T, 1]
    payout_ratio: tf.Tensor             # [B, T, 1]

    # --- Liquidity policy ---
    min_cash_ratio: tf.Tensor           # [B, T, 1]

    # --- Financing policy ---
    lt_share_for_capex: tf.Tensor       # [B, T, 1]
    st_invest_spread: tf.Tensor         # [B, T, 1]
    debt_spread: tf.Tensor              # [B, T, 1], # spread over real rates for debt cost

    # --- Operating/efficiency schedules (parameters) ---
    # Working capital days policies [B, T, 1]
    dso_days: tf.Tensor
    dpo_days: tf.Tensor
    dio_days: tf.Tensor
    # Opex ratio and depreciation schedule [B, T, 1]
    opex_ratio: tf.Tensor
    depreciation_rate: tf.Tensor
    # Optional gross margin policy (if provided can derive COGS instead of simulating directly)
    gross_margin: Optional[tf.Tensor] = None  # [B, T, 1]

    # --- Learnable coefficients (fallback derivations) ---
    # Misc period length
    period_days: float = 365.0/4.0
    # Optional override metric for liquidity (placed after required non-default args for dataclass ordering)
    cash_coverage: Optional[tf.Tensor] = None  # [B, T, 1]
    # AOCI component sensitivities (units depend on data provider sign conventions)
    kappa_fx: Optional[tf.Tensor] = None             # [B, T, 1]
    kappa_unrealized: Optional[tf.Tensor] = None     # [B, T, 1]
    kappa_other: Optional[tf.Tensor] = None          # [B, T, 1]
    # Goodwill premium ratio to apply to aggregate investing CF (negative implies acquisitions)
    premium_ratio_goodwill: Optional[tf.Tensor] = None  # [B, T, 1]
    # Other NCA linear model coefficients
    beta1_capex: Optional[tf.Tensor] = None          # [B, T, 1]
    beta2_net_invest: Optional[tf.Tensor] = None     # [B, T, 1]
    # Capital stock sensitivity to net common stock issuance (issuance grows par value a bit)
    gamma_capital_stock: Optional[tf.Tensor] = None  # [B, T, 1]
    # Lease schedule parameters (additions and schedule rates)
    lease_addition_capex_coeff: Optional[tf.Tensor] = None   # [B, T, 1]
    lease_addition_sales_coeff: Optional[tf.Tensor] = None   # [B, T, 1]
    lease_avg_remaining_term: Optional[tf.Tensor] = None     # [B, T, 1] (quarters)
    lease_principal_payment_rate: Optional[tf.Tensor] = None # [B, T, 1] (0..1)
    lease_termination_sales_coeff: Optional[tf.Tensor] = None# [B, T, 1] (>=0)
    # Other current assets delta model (fallback)
    omega_oca_sales: Optional[tf.Tensor] = None      # [B, T, 1]
    omega_oca_opex: Optional[tf.Tensor] = None       # [B, T, 1]
    # Other non-current liabilities delta model (fallback)
    psi_oncl_deferred_tax: Optional[tf.Tensor] = None   # [B, T, 1]
    psi_oncl_other_nc: Optional[tf.Tensor] = None       # [B, T, 1]
    # Paid-in capital sensitivity to long-term debt changes
    k_pi: Optional[tf.Tensor] = None               # [B, T, 1]

@dataclass
class DriversWMT:
    # Behavioral/exogenous drivers to feed the structural layer [B, T, 1]

    # --- Financials (income statement) lines ---
    sales: tf.Tensor               # [B, T, 1], Total Revenue
    cogs: tf.Tensor                # [B, T, 1], Cost of Revenue

    # --- Balance sheet lines (paths, passed-through, not plugs) ---
    # Preferred delta drivers for leases (if absent, engine uses policy schedule)
    change_in_current_capital_lease_obligation: Optional[tf.Tensor] = None  # [B, T, 1]
    change_in_long_term_capital_lease_obligation: Optional[tf.Tensor] = None # [B, T, 1]

    # --- Cash flow lines (deltas and features) ---
    # Capex path (if not provided, engine derives base from depreciation)
    capex: Optional[tf.Tensor] = None    # [B, T, 1], Capital Expenditures from Cash Flow Statement
    # Working-capital related deltas
    change_in_accrued_expenses: Optional[tf.Tensor] = None    # [B, T, 1]
    change_in_tax_payable: Optional[tf.Tensor] = None         # [B, T, 1]
    cash_dividends_paid: Optional[tf.Tensor] = None           # [B, T, 1]
    # Optional deltas overriding level pass-throughs
    # goodwill and other NCA deltas removed from Drivers; derived internally from CF features
    # AOCI total delta (single independent driver). For history we typically
    # construct this from the vendor "Gains Losses Not Affecting Retained Earnings"
    # line as ΔAOCI_t = AOCI_t - AOCI_{t-1}.
    change_in_aoci: Optional[tf.Tensor] = None                 # [B, T, 1]
    # Features used to derive deltas when not provided
    effect_of_exchange_rate_changes: Optional[tf.Tensor] = None        # [B, T, 1]
    gain_loss_on_investment_securities: Optional[tf.Tensor] = None     # [B, T, 1]
    deferred_income_tax: Optional[tf.Tensor] = None                    # [B, T, 1]
    other_non_cash_items: Optional[tf.Tensor] = None                   # [B, T, 1]
    # Minority interest delta (single-driver design)
    change_in_minority_interest: Optional[tf.Tensor] = None    # [B, T, 1]
    # Investing cash flow features
    # Aggregate net acquisitions and investing CF (canonical driver)
    aggregate_invest: Optional[tf.Tensor] = None                       # [B, T, 1]
    # Equity financing (net issuance/repurchase)
    net_common_stock_issuance: Optional[tf.Tensor] = None      # [B, T, 1]

@dataclass
class PrevStateWMT:
    # End-of-last-period balances (used for interest on beginning balances) [B, 1]

    # --- Assets (required) ---
    cash: tf.Tensor
    st_investments: tf.Tensor
    ar: tf.Tensor
    inventory: tf.Tensor
    net_ppe: tf.Tensor

    # --- Liabilities (required) ---
    st_debt: tf.Tensor
    lt_debt: tf.Tensor
    ap: tf.Tensor

    # --- Equity (required) ---
    equity: tf.Tensor  # parent equity excluding minority interest

    # --- Assets (optional extensions) ---
    other_current_assets: Optional[tf.Tensor] = None
    goodwill_intangibles: Optional[tf.Tensor] = None
    other_non_current_assets: Optional[tf.Tensor] = None

    # --- Liabilities (optional extensions) ---
    accrued_expenses: Optional[tf.Tensor] = None
    tax_payable: Optional[tf.Tensor] = None
    other_non_current_liabilities: Optional[tf.Tensor] = None
    current_capital_lease_obligation: Optional[tf.Tensor] = None
    long_term_capital_lease_obligation: Optional[tf.Tensor] = None
    dividends_payable: Optional[tf.Tensor] = None

    # --- Equity components (optional) ---
    # AOCI total balance; for WMT we map this from the vendor line
    # "Gains Losses Not Affecting Retained Earnings".
    aoci: Optional[tf.Tensor] = None
    minority_interest: Optional[tf.Tensor] = None
    capital_stock: Optional[tf.Tensor] = None

    # Optional decomposition (if None we infer retained earnings = equity, paid-in capital = 0)
    retained_earnings: Optional[tf.Tensor] = None
    paid_in_capital: Optional[tf.Tensor] = None

@dataclass
class StatementsWMT:
    # All tensors shaped [B, T, 1]

    # --- Income statement ---
    sales: tf.Tensor
    cogs: tf.Tensor
    gross_profit: tf.Tensor
    opex: tf.Tensor
    ebit: tf.Tensor
    interest_income: tf.Tensor
    interest_expense: tf.Tensor
    ebt: tf.Tensor
    taxes: tf.Tensor
    net_income: tf.Tensor

    # --- Shareholder distributions ---
    dividends: tf.Tensor

    # --- Cash flow (minimal for BS roll-forward) ---
    capex: tf.Tensor
    depreciation: tf.Tensor
    wc_change: tf.Tensor

    # --- Balance Sheet (end of period) ---
    # Assets
    cash: tf.Tensor
    st_investments: tf.Tensor
    ar: tf.Tensor
    inventory: tf.Tensor
    other_current_assets: tf.Tensor
    net_ppe: tf.Tensor
    goodwill_intangibles: tf.Tensor
    other_non_current_assets: tf.Tensor
    # Liabilities
    st_debt: tf.Tensor
    lt_debt: tf.Tensor
    ap: tf.Tensor
    accrued_expenses: tf.Tensor
    tax_payable: tf.Tensor
    dividends_payable: tf.Tensor
    other_non_current_liabilities: tf.Tensor
    current_capital_lease_obligation: tf.Tensor
    long_term_capital_lease_obligation: tf.Tensor
    # Equity and related
    equity: tf.Tensor
    retained_earnings: tf.Tensor
    paid_in_capital: tf.Tensor
    capital_stock: tf.Tensor
    aoci: tf.Tensor
    minority_interest: tf.Tensor
    # add a common stock prev
    # add common stock equity component
    # add gains losses not affecing retained earnings
    # add total equity gross minority interest

    @property
    def assets(self) -> tf.Tensor:
        return (
            self.cash + self.st_investments + self.ar + self.inventory + self.other_current_assets + self.net_ppe + self.goodwill_intangibles + self.other_non_current_assets
        )

    @property
    def liab_plus_equity(self) -> tf.Tensor:
        return (
            self.st_debt + self.current_capital_lease_obligation +
            self.lt_debt + self.long_term_capital_lease_obligation +
            self.ap + self.accrued_expenses + self.tax_payable + self.dividends_payable +
            self.other_non_current_liabilities +
            (self.equity + self.capital_stock + self.aoci) + self.minority_interest
        )

    @property
    def core_assets(self) -> tf.Tensor:
        """Core modeled assets excluding exogenous extensions (for strict identity sanity)."""
        return self.cash + self.st_investments + self.ar + self.inventory + self.net_ppe

    @property
    def core_liab_plus_equity(self) -> tf.Tensor:
        """Core modeled liabilities + equity excluding exogenous extensions.

        Note: This is the legacy core subset used for regression sanity only.
        Prefer the full expanded identity for correctness.
        """
        return self.st_debt + self.lt_debt + self.ap + self.equity

    @property
    def identity_gap(self) -> tf.Tensor:
        return tf.abs(self.assets - self.liab_plus_equity)