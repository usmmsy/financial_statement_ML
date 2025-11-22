from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import tensorflow as tf


Tensor = tf.Tensor  # [B, T, 1] unless specified; PrevState fields are [B, 1]

#
# === BALANCE-SHEET DATA CLASSES (faithful to timing + FIFO) ===
#

@dataclass
class Policies:
    # --- Pricing / rates ---
    fixed_asset: Tensor        # [B,T,1] fixed asset value
    depr_rate: Tensor        # [B,T,1] straight-line depreciation rate (for NFA roll)
    inflation: Tensor          # [B,T,1] general inflation (for unit-cost indexation)
    tax_rate: Tensor           # [B,T,1] corporate tax rate
    payout_ratio: Tensor       # [B,T,1] fraction of NI_t paid as dividend at t+1
    real_rate: Tensor          # [B,T,1] real ST rate

    # --- Operational policy levers ---
    advertisement_sales_ratio: Tensor  # [B,T,1] ad spend as % of sales (for sales boost modeling)
    inventory_volume_ratio: Tensor  # [B,T,1] inventory in units per unit sold (for FIFO layers)

    # --- Timing rails: sales ---
    sales_adv_share: Tensor    # [B,T,1] received at t-1 for sales of t (customer advances)
    sales_credit_share: Tensor # [B,T,1] accrued as AR at t, collected at t+1
    # same-year cash share is implied: 1 - adv - credit

    # --- Timing rails: purchases ---
    purch_adv_share: Tensor    # [B,T,1] advance paid at t for purchases delivered at t+1 (supplier advances)
    purch_credit_share: Tensor # [B,T,1] AP accrued at t, paid at t+1
    # same-year cash share is implied: 1 - adv - credit

    # --- Financing rule (deficit) ---
    deficit_debt_share: Tensor  # [B,T,1] share of deficit funded with debt (rest is equity)

    min_cash_ratio: Tensor    # [B,T,1] target minimum cash to sales ratio

    # Optional policy overrides (kept for compatibility; may be None):
    lt_rate: Optional[Tensor] = None          # [B,T,1]
    st_borrow_rate: Optional[Tensor] = None   # [B,T,1]
    st_invest_rate: Optional[Tensor] = None   # [B,T,1]
    opex_ratio: Optional[Tensor] = None       # [B,T,1]
    repurchase_share_of_depr: Optional[Tensor] = None  # [B,T,1] (capex replacement heuristic)

    # Convenience: subscriptable by batch/time
    def slice_t(self, t: int) -> "Policies":
        def sel(v):
            if v is None: return None
            return v[:, t:t+1, :]
        return Policies(
            inflation=sel(self.inflation), real_rate=sel(self.real_rate), tax_rate=sel(self.tax_rate),
            payout_ratio=sel(self.payout_ratio),
            advertisement_sales_ratio=sel(self.advertisement_sales_ratio), inventory_volume_ratio=sel(self.inventory_volume_ratio),
            sales_adv_share=sel(self.sales_adv_share), sales_credit_share=sel(self.sales_credit_share),
            purch_adv_share=sel(self.purch_adv_share), purch_credit_share=sel(self.purch_credit_share),
            deficit_debt_share=sel(self.deficit_debt_share), min_cash_ratio=sel(self.min_cash_ratio),
            lt_rate=sel(self.lt_rate) if self.lt_rate is not None else None,
            st_borrow_rate=sel(self.st_borrow_rate) if self.st_borrow_rate is not None else None,
            st_invest_rate=sel(self.st_invest_rate) if self.st_invest_rate is not None else None,
            opex_ratio=sel(self.opex_ratio) if self.opex_ratio is not None else None,
            repurchase_share_of_depr=sel(self.repurchase_share_of_depr) if self.repurchase_share_of_depr is not None else None,
        )

    # Optional numpy-like indexing (batch/time only, not features)
    def __getitem__(self, idx) -> "Policies":
        def ix(v):
            if v is None: return None
            return v[idx]
        return Policies(
            inflation=ix(self.inflation), real_rate=ix(self.real_rate), tax_rate=ix(self.tax_rate),
            payout_ratio=ix(self.payout_ratio), advertisement_sales_ratio=ix(self.advertisement_sales_ratio),
            inventory_volume_ratio=ix(self.inventory_volume_ratio),
            sales_adv_share=ix(self.sales_adv_share), sales_credit_share=ix(self.sales_credit_share),
            purch_adv_share=ix(self.purch_adv_share), purch_credit_share=ix(self.purch_credit_share),
            deficit_debt_share=ix(self.deficit_debt_share), min_cash_ratio=ix(self.min_cash_ratio),
            lt_rate=ix(self.lt_rate) if self.lt_rate is not None else None,
            st_borrow_rate=ix(self.st_borrow_rate) if self.st_borrow_rate is not None else None,
            st_invest_rate=ix(self.st_invest_rate) if self.st_invest_rate is not None else None,
            opex_ratio=ix(self.opex_ratio) if self.opex_ratio is not None else None,
            repurchase_share_of_depr=ix(self.repurchase_share_of_depr) if self.repurchase_share_of_depr is not None else None,
        )


@dataclass
class Drivers:
    # Operating drivers (deterministic; no DSO/DPO/DIO “plugs”)
    price: Tensor             # [B,T,1]
    volume: Tensor            # [B,T,1] (units sold)
    unit_cost: Tensor         # [B,T,1] (purchase cost per unit for FIFO layer)
    opex: Optional[Tensor] = None  # [B,T,1] optional direct opex; else use policy ratio


@dataclass
class PrevState:
    # Period t-1 end balances, shape [B,1,1]
    cash: Tensor
    st_investments: Tensor
    st_debt: Tensor
    lt_debt: Tensor
    ar: Tensor
    ap: Tensor
    inventory_units: Tensor  # inventory in units (FIFO)
    inventory_value: Tensor  # inventory value (FIFO valuation)
    nfa: Tensor              # net fixed assets (book)
    equity: Tensor

    # Rails stocks (explicit timing accounts)
    advances_received: Tensor   # customer advances (deferred revenue) [B,1,1]
    advances_paid: Tensor       # supplier advances (other current asset) [B,1,1]

    # Dividends payable (div declared on NI_{t-1}, paid at t)
    dividends_payable: Tensor


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
