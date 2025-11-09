from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple, Union
import tensorflow as tf

Index = Union[int, slice, tf.Tensor, list]

@dataclass
class Statements:
    """
    Model-generated financial statements for period t.
    
    All tensors have shape [B, T, 1] unless otherwise noted:
        1. B: batch dimension (e.g., different companies / scenarios)
        2. T: time steps in the forecast horizon
    """

    # Income statement lines
    sales: tf.Tensor # revenue
    cogs: tf.Tensor # cost of goods sold
    opex: tf.Tensor # operating expenses
    ebit: tf.Tensor # earnings before interest and taxes
    interest: tf.Tensor # interest expense
    tax: tf.Tensor # income tax expense
    net_income: tf.Tensor # after-tax income

    # Balance sheet lines (end-of-period balances)
    cash: tf.Tensor # cash and cash equivalents
    ar: tf.Tensor # accounts receivable
    ap: tf.Tensor # accounts payable
    inventory:tf.Tensor # inventory
    st_investments:tf.Tensor # short-term investments
    st_debt:tf.Tensor # short-term debt
    lt_debt:tf.Tensor # long-term debt
    nfa:tf.Tensor # net fixed assets
    equity:tf.Tensor # shareholders' equity

    # Cash budget support
    ncb: tf.Tensor # net cash budget

    @property
    def assets(self) -> tf.Tensor:
        """
        Total assets
        cash + ar + inventory + st_investments + nfa
        shape: [B, T, 1]
        """
        
        return (
            self.cash 
            + self.ar 
            + self.inventory 
            + self.st_investments 
            + self.nfa
        )

    @property
    def liab_plus_equity(self) -> tf.Tensor:
        """
        Total liabilities and equity
        ap + st_debt + lt_debt + equity
        shape: [B, T, 1]
        """

        return (
            self.ap 
            + self.st_debt 
            + self.lt_debt 
            + self.equity
        )
    
    def balance_sheet_view(self) -> dict:
        """
        Returns a dictionary representation of the balance sheet lines.
        Useful for converting to DataFrames or other formats.
        """

        return {
            "cash": self.cash,
            "short_term_investments": self.st_investments,
            "accounts_receivable": self.ar,
            "accounts_payable": self.ap,
            "inventory": self.inventory,
            "short_term_debt": self.st_debt,
            "long_term_debt": self.lt_debt,
            "net_fixed_assets": self.nfa,
            "equity": self.equity,
            "total_assets": self.assets,
            "total_liabilities_and_equity": self.liab_plus_equity,
        }
    
@dataclass
class Drivers:
    """
    Behavioral drivers predicted by DriverHead.
    
    All tensors have shape [B, T, 1] unless otherwise noted:
    
    These are NOT plugs but represent managerial / operational decisions that
    feeds the deterministic accounting logic.
    """
    price: tf.Tensor # selling price level / ASP
    volume: tf.Tensor # sales volume
    dso_days: tf.Tensor # days sales outstanding
    dpo_days: tf.Tensor # days payable outstanding
    dio_days: tf.Tensor # days inventory outstanding
    capex: tf.Tensor # capital expenditures
    stlt_split: tf.Tensor # fraction of new debt that is LT debt

@dataclass
class Policies:
    """
    Exogenous policy / assumption inputs, not learned.
    
    All tensors have shape [B, T, 1].
    
    These encode management policies, macroeconomic assumptions or treasury constraints.
    """

    inflation: tf.Tensor # CPI path
    real_rate: tf.Tensor # real short-term rate path
    tax_rate: tf.Tensor # statutory/base tax rate path
    min_cash_ratio: tf.Tensor # target minimum cash to sales ratio
    payout_ratio: tf.Tensor # dividend payout ratio

    # Optional overrides / richer policy levers (all [B, T, 1] if present)
    lt_rate: Optional[tf.Tensor] = None          # long-term borrowing rate
    opex_ratio: Optional[tf.Tensor] = None       # opex as % of sales
    depreciation_rate: Optional[tf.Tensor] = None
    cost_share: Optional[tf.Tensor] = None       # COGS share for inventory calc
    st_rate: Optional[tf.Tensor] = None   # ST debt rate
    st_invest_rate: Optional[tf.Tensor] = None   # ST invest yield
    cash_coverage: Optional[tf.Tensor] = None    # min-cash coverage of (COGS+Opex+Tax)

    # ---- slicing core: supports [batch, time, feature] ----
    def __getitem__(self, sel: Union[Index, Tuple[Optional[Index], Optional[Index], Optional[Index]]]) -> "Policies":
        # Normalize sel to a 3-tuple (b_sel, t_sel, f_sel)
        if not isinstance(sel, tuple):
            # Single index means "time" for ergonomic  policies[t]  or  policies[:t]
            b_sel, t_sel, f_sel = None, sel, None
        else:
            # pad/trim to 3
            s = list(sel) + [None, None, None]
            b_sel, t_sel, f_sel = s[0], s[1], s[2]

        def _normalize(idx: Optional[Index], axis_len: Optional[int] = None) -> Any:
            # None / ":" -> full slice
            if idx is None:
                return slice(None)
            # ints -> keep dim via i:i+1
            if isinstance(idx, int):
                return slice(idx, idx + 1, 1)
            # slices -> as-is (we’ll special-case step later)
            if isinstance(idx, slice):
                return idx
            # lists/1D tensors/ndarrays -> use as gather indices
            return idx  # handled in _apply on the tensor
        b_sel = _normalize(b_sel)
        t_sel = _normalize(t_sel)
        f_sel = _normalize(f_sel)

        def _apply(x: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
            if x is None:
                return None

            # Apply batch selector (axis=0)
            y = _apply_one_axis(x, b_sel, axis=0)
            # Apply time selector (axis=1)
            y = _apply_one_axis(y, t_sel, axis=1)
            # Apply feature selector (axis=2)
            y = _apply_one_axis(y, f_sel, axis=2)
            return y

        def _apply_one_axis(v: tf.Tensor, idx: Any, axis: int) -> tf.Tensor:
            # slice(None) -> no-op
            if isinstance(idx, slice):
                # If step is None or 1, do direct slice. Else, gather stepped indices.
                step = idx.step
                start = idx.start
                stop = idx.stop
                if step in (None, 1):
                    start = 0 if start is None else start
                    # stop can be None (=> till end) which tf slice supports
                    slicers = [slice(None), slice(None), slice(None)]
                    slicers[axis] = slice(start, stop, 1)
                    return v[tuple(slicers)]
                else:
                    # stepped slice -> gather
                    start = 0 if start is None else start
                    # build indices tensor
                    if stop is None:
                        # infer length along this axis
                        axis_len = tf.shape(v)[axis]
                        # tf.range requires tensors
                        idxs = tf.range(start, axis_len, delta=step)
                    else:
                        idxs = tf.range(start, stop, delta=step)
                    return tf.gather(v, idxs, axis=axis)

            # list / tensor indices -> gather
            if isinstance(idx, (list, tuple)) or tf.is_tensor(idx):
                idxs = tf.convert_to_tensor(idx, dtype=tf.int32)
                return tf.gather(v, idxs, axis=axis)

            # Shouldn’t get here (we normalized ints -> slices above)
            return v

        kw: Dict[str, Optional[tf.Tensor]] = {f.name: _apply(getattr(self, f.name)) for f in fields(self)}
        return Policies(**kw)

    # ---- ergonomic time helpers (sugar) ----
    def at(self, t: int) -> "Policies":
        """Equivalent to policies[:, t, :], keeping dims -> [B, 1, 1]."""
        return self[:, t, :]

    def time_window(self, start: int, stop: int, step: Optional[int] = None) -> "Policies":
        """Equivalent to policies[:, start:stop:step, :]."""
        return self[:, slice(start, stop, step), :]

@dataclass
class PrevState:
    """
    End-of-period state from t-1, used as the starting point for period t.
    
    Each tensor has shape [B, 1].
    (Single time slice: the last known balance sheet we roll forward from.)

    StructuralLayer uses this to:
        1. compute interest on prior debt / investments
        2. roll cash forward
        3. roll debt forward
        4. rolll equity via retained earnings
        5. roll NFA via capex and depreciation
    """

    cash: tf.Tensor
    st_investments: tf.Tensor
    st_debt: tf.Tensor
    lt_debt: tf.Tensor
    ar: tf.Tensor
    ap: tf.Tensor
    inventory: tf.Tensor
    nfa: tf.Tensor
    equity: tf.Tensor