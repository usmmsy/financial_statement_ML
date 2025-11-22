from typing import List
import tensorflow as tf
from .types_wmt import PoliciesWMT, DriversWMT, PrevStateWMT, StatementsWMT

# --- helpers (TensorFlow) ---
def fisher_nominal_tf(real_rate: tf.Tensor, inflation: tf.Tensor) -> tf.Tensor:
    """Fisher relation: (1+r_nom) = (1+r_real)*(1+pi) - 1"""
    return (1.0 + tf.nn.relu(real_rate)) * (1.0 + inflation) - 1.0

def days_to_balance_tf(days: tf.Tensor, flow: tf.Tensor, period_days: float) -> tf.Tensor:
    """
    Convert policy in days to an ending balance given the period flow:
        balance ≈ (days / period_days) * relu(flow)
    """
    return tf.nn.relu(days) / period_days * tf.nn.relu(flow)


class StructuralLayer(tf.keras.layers.Layer):
    """Deterministic roll-forward (Vélez–Pareja) for WMT.

    Policies are provided at call time rather than construction so they can be
    treated as trainable in calibration loops.
    """

    def __init__(self, hard_identity_check: bool = False, identity_tol: float = 1e-6, collect_diagnostics: bool = False):
        super().__init__()
        self.hard_identity_check = hard_identity_check
        self.identity_tol = float(identity_tol)
        self.collect_diagnostics = collect_diagnostics
        self.last_diagnostics = None

    def call(self, drivers: DriversWMT, policies: PoliciesWMT, prev: PrevStateWMT, training: bool = False) -> StatementsWMT:
        # Ensure tensors and dtypes; expect shapes [B, T, 1] for series, [B, 1] for prev
        sales = tf.cast(drivers.sales, tf.float32)
        cogs = tf.cast(drivers.cogs, tf.float32)
        capex_path = drivers.capex  # may be None

        B = tf.shape(sales)[0]
        T = tf.shape(sales)[1]

        # Policies
        p = policies
        inflation = tf.cast(p.inflation, tf.float32)
        real_st_rate = tf.cast(p.real_st_rate, tf.float32)
        real_lt_rate = tf.cast(p.real_lt_rate, tf.float32)
        tax_rate = tf.cast(p.tax_rate, tf.float32)
        min_cash_ratio = tf.cast(p.min_cash_ratio, tf.float32)
        cash_cov = None if p.cash_coverage is None else tf.cast(p.cash_coverage, tf.float32)
        lt_share_for_capex = tf.cast(p.lt_share_for_capex, tf.float32)
        st_invest_spread = tf.cast(p.st_invest_spread, tf.float32)
        debt_spread = tf.cast(p.debt_spread, tf.float32)
        payout_ratio = tf.cast(p.payout_ratio, tf.float32)
        # Moved ratios/schedules from Drivers to Policies
        dso_days = tf.cast(p.dso_days, tf.float32)
        dpo_days = tf.cast(p.dpo_days, tf.float32)
        dio_days = tf.cast(p.dio_days, tf.float32)
        opex_ratio = tf.cast(p.opex_ratio, tf.float32)
        depreciation_rate = tf.cast(p.depreciation_rate, tf.float32)
        gross_margin = tf.cast(p.gross_margin, tf.float32) if getattr(p, 'gross_margin', None) is not None else None
        period_days = float(p.period_days)
        # Optional coefficients with safe fallbacks
        kappa_fx = tf.cast(getattr(p, 'kappa_fx', None), tf.float32) if getattr(p, 'kappa_fx', None) is not None else tf.zeros_like(inflation)
        kappa_unr = tf.cast(getattr(p, 'kappa_unrealized', None), tf.float32) if getattr(p, 'kappa_unrealized', None) is not None else tf.zeros_like(inflation)
        kappa_other = tf.cast(getattr(p, 'kappa_other', None), tf.float32) if getattr(p, 'kappa_other', None) is not None else tf.zeros_like(inflation)
        premium_gw = tf.cast(getattr(p, 'premium_ratio_goodwill', None), tf.float32) if getattr(p, 'premium_ratio_goodwill', None) is not None else tf.zeros_like(inflation)
        beta1 = tf.cast(getattr(p, 'beta1_capex', None), tf.float32) if getattr(p, 'beta1_capex', None) is not None else tf.zeros_like(inflation)
        beta2 = tf.cast(getattr(p, 'beta2_net_invest', None), tf.float32) if getattr(p, 'beta2_net_invest', None) is not None else tf.zeros_like(inflation)
        gamma_cap = tf.cast(getattr(p, 'gamma_capital_stock', None), tf.float32) if getattr(p, 'gamma_capital_stock', None) is not None else tf.zeros_like(inflation)
        # Lease schedule policy parameters: we retain only capex- and sales-based additions
        lease_addition_capex_coeff = tf.cast(getattr(p, 'lease_addition_capex_coeff', None), tf.float32) if getattr(p, 'lease_addition_capex_coeff', None) is not None else tf.zeros_like(inflation)
        lease_addition_sales_coeff = tf.cast(getattr(p, 'lease_addition_sales_coeff', None), tf.float32) if getattr(p, 'lease_addition_sales_coeff', None) is not None else tf.zeros_like(inflation)
        lease_avg_remaining_term = tf.cast(getattr(p, 'lease_avg_remaining_term', None), tf.float32) if getattr(p, 'lease_avg_remaining_term', None) is not None else tf.ones_like(inflation)
        lease_principal_payment_rate = tf.cast(getattr(p, 'lease_principal_payment_rate', None), tf.float32) if getattr(p, 'lease_principal_payment_rate', None) is not None else tf.zeros_like(inflation)
        lease_termination_sales_coeff = tf.cast(getattr(p, 'lease_termination_sales_coeff', None), tf.float32) if getattr(p, 'lease_termination_sales_coeff', None) is not None else tf.zeros_like(inflation)
        # Fallbacks for other current assets and other non-current liabilities
        omega_oca_sales = tf.cast(getattr(p, 'omega_oca_sales', None), tf.float32) if getattr(p, 'omega_oca_sales', None) is not None else tf.zeros_like(inflation)
        omega_oca_opex = tf.cast(getattr(p, 'omega_oca_opex', None), tf.float32) if getattr(p, 'omega_oca_opex', None) is not None else tf.zeros_like(inflation)
        psi_oncl_deferred_tax = tf.cast(getattr(p, 'psi_oncl_deferred_tax', None), tf.float32) if getattr(p, 'psi_oncl_deferred_tax', None) is not None else tf.zeros_like(inflation)
        psi_oncl_other_nc = tf.cast(getattr(p, 'psi_oncl_other_nc', None), tf.float32) if getattr(p, 'psi_oncl_other_nc', None) is not None else tf.zeros_like(inflation)

        k_pi = tf.cast(getattr(p, 'k_pi', None), tf.float32) if getattr(p, 'k_pi', None) is not None else tf.zeros_like(inflation)

        # Rates per period
        rf_q = fisher_nominal_tf(real_st_rate, inflation)
        st_invest_rate = rf_q + st_invest_spread
        st_borrow_rate = rf_q + debt_spread
        lt_borrow_rate = fisher_nominal_tf(real_lt_rate, inflation) + debt_spread

        # Initial state [B, 1, 1]
        def _expand_prev(x: tf.Tensor) -> tf.Tensor:
            x = tf.cast(x, tf.float32)
            return tf.reshape(x, [tf.shape(sales)[0], 1, 1])

        def _prev_or_zero(opt: tf.Tensor) -> tf.Tensor:
            return opt if opt is not None else tf.zeros_like(prev.cash)

        cash0 = _expand_prev(prev.cash)
        sti0 = _expand_prev(prev.st_investments)
        stdebt0 = _expand_prev(prev.st_debt)
        ltdebt0 = _expand_prev(prev.lt_debt)
        ar0 = _expand_prev(prev.ar)
        ap0 = _expand_prev(prev.ap)
        inv0 = _expand_prev(prev.inventory)
        ppe0 = _expand_prev(prev.net_ppe)
        equity0 = _expand_prev(prev.equity)
        other_cur0 = _expand_prev(_prev_or_zero(prev.other_current_assets))
        goodwill0 = _expand_prev(_prev_or_zero(prev.goodwill_intangibles))
        other_nca0 = _expand_prev(_prev_or_zero(prev.other_non_current_assets))
        accr_exp0 = _expand_prev(_prev_or_zero(prev.accrued_expenses))
        tax_pay0 = _expand_prev(_prev_or_zero(prev.tax_payable))
        other_ncl0 = _expand_prev(_prev_or_zero(prev.other_non_current_liabilities))
        # AOCI previous total balance
        aoci0 = _expand_prev(_prev_or_zero(getattr(prev, 'aoci', None)))
        minority0 = _expand_prev(_prev_or_zero(prev.minority_interest))
        cur_lease0 = _expand_prev(_prev_or_zero(prev.current_capital_lease_obligation))
        lt_lease0 = _expand_prev(_prev_or_zero(prev.long_term_capital_lease_obligation))
        div_pay0 = _expand_prev(_prev_or_zero(prev.dividends_payable))
        # other_payables removed (redundant decomposition of AP)
        cap_stock0 = _expand_prev(_prev_or_zero(prev.capital_stock))
        retained0 = _expand_prev(prev.retained_earnings if prev.retained_earnings is not None else prev.equity)
        paidin0 = _expand_prev(prev.paid_in_capital if prev.paid_in_capital is not None else tf.zeros_like(prev.equity))

        # Time-major helpers (deterministic Python loop)
        tm = lambda x: tf.transpose(x, [1, 0, 2])
        has_cov = cash_cov is not None
        sales_tm = tm(sales); cogs_tm = tm(cogs); dso_tm = tm(dso_days); dpo_tm = tm(dpo_days); dio_tm = tm(dio_days)
        stinv_rate_tm = tm(st_invest_rate); stbor_rate_tm = tm(st_borrow_rate); ltbor_rate_tm = tm(lt_borrow_rate)
        ltshare_tm = tm(lt_share_for_capex)
        # kappa_* retained for backward compatibility but no longer used for AOCI roll-forward
        kappa_fx_tm = tm(kappa_fx); kappa_unr_tm = tm(kappa_unr); kappa_other_tm = tm(kappa_other)
        premium_gw_tm = tm(premium_gw); beta1_tm = tm(beta1); beta2_tm = tm(beta2)
        gamma_cap_tm = tm(gamma_cap)
        # Time-major lease schedule coefficients (capex and sales only)
        lease_addition_capex_coeff_tm = tm(lease_addition_capex_coeff); lease_addition_sales_coeff_tm = tm(lease_addition_sales_coeff)
        lease_avg_term_tm = tm(lease_avg_remaining_term)
        lease_principal_rate_tm = tm(lease_principal_payment_rate)
        lease_termination_sales_coeff_tm = tm(lease_termination_sales_coeff)
        omega_oca_sales_tm = tm(omega_oca_sales); omega_oca_opex_tm = tm(omega_oca_opex)
        psi_oncl_deferred_tax_tm = tm(psi_oncl_deferred_tax); psi_oncl_other_nc_tm = tm(psi_oncl_other_nc)
        capex_tm = tm(capex_path) if capex_path is not None else tf.fill([T, B, 1], tf.constant(float('nan'), tf.float32))
        opex_ratio_tm = tm(opex_ratio); depr_rate_tm = tm(depreciation_rate); tax_rate_tm = tm(tax_rate)
        gross_margin_tm = tm(gross_margin) if gross_margin is not None else None
        mincash_tm = tm(min_cash_ratio); payout_tm = tm(payout_ratio); cashcov_tm = tm(cash_cov) if has_cov else None

        def s3(x):
            return tf.expand_dims(x, -1) if x.shape.rank == 2 else x

        # Carry state
        cash_beg = cash0; sti_beg = sti0; st_beg = stdebt0; lt_beg = ltdebt0
        ar_beg = ar0; ap_beg = ap0; inv_beg = inv0; ppe_beg = ppe0
        equity_beg = equity0; retained_beg = retained0; paid_in_beg = paidin0

        # final list and downstream stack squeeze error. Keep this in sync with vals construction.
        # Removing treasury_stock from outputs; total count decreases by 1
        outs = [[] for _ in range(37)]
        Tn = int(tf.shape(sales_tm)[0].numpy())  # eager loop length
        diag_rows = [] if self.collect_diagnostics else None

        # Track previous sales to compute delta sales for lease fallback models
        sales_prev = None

        for t in range(Tn):
            t_sales = s3(sales_tm[t]); t_dso = s3(dso_tm[t]); t_dpo = s3(dpo_tm[t]); t_dio = s3(dio_tm[t])
            # Derive COGS from gross margin policy if provided; else use driver COGS path
            if gross_margin_tm is not None:
                t_gm = s3(gross_margin_tm[t])
                t_cogs = (1.0 - tf.clip_by_value(t_gm, 0.0, 0.95)) * t_sales
            else:
                t_cogs = s3(cogs_tm[t])
            t_stinv_rate = s3(stinv_rate_tm[t]); t_stbor_rate = s3(stbor_rate_tm[t]); t_ltbor_rate = s3(ltbor_rate_tm[t])
            t_lt_share = s3(ltshare_tm[t]); t_capex = s3(capex_tm[t]); t_opex_ratio = s3(opex_ratio_tm[t])
            t_depr_rate = s3(depr_rate_tm[t]); t_tax_rate = s3(tax_rate_tm[t]); t_min_cash_ratio = s3(mincash_tm[t])
            t_cash_cov = s3(cashcov_tm[t]) if has_cov else None; t_payout = s3(payout_tm[t])
            t_kfx = s3(kappa_fx_tm[t]); t_kunr = s3(kappa_unr_tm[t]); t_koth = s3(kappa_other_tm[t])
            t_prem = s3(premium_gw_tm[t]); t_b1 = s3(beta1_tm[t]); t_b2 = s3(beta2_tm[t])
            t_lease_add_capex = s3(lease_addition_capex_coeff_tm[t]); t_lease_add_sales = s3(lease_addition_sales_coeff_tm[t])
            t_lease_avg_term = s3(lease_avg_term_tm[t]); t_lease_princ_rate = s3(lease_principal_rate_tm[t])
            t_lease_term_sales = s3(lease_termination_sales_coeff_tm[t])

            # Compute delta sales for this period (zeros for first period)
            if sales_prev is None:
                delta_sales = tf.zeros_like(t_sales)
            else:
                delta_sales = t_sales - sales_prev
            sales_prev = t_sales

            opex = t_opex_ratio * t_sales
            depreciation = t_depr_rate * tf.nn.relu(ppe_beg)
            interest_income = t_stinv_rate * tf.nn.relu(sti_beg)
            # Include lease obligations in interest expense (current + long term portions)
            interest_expense = (
                t_stbor_rate * tf.nn.relu(st_beg + cur_lease0) +
                t_ltbor_rate * tf.nn.relu(lt_beg + lt_lease0)
            )
            ebit = t_sales - t_cogs - opex - depreciation
            ebt = ebit + interest_income - interest_expense
            taxes = t_tax_rate * tf.nn.relu(ebt)
            net_income = ebt - taxes
            dividends = t_payout * tf.nn.relu(net_income)

            ar_end = days_to_balance_tf(t_dso, t_sales, period_days)
            ap_end = days_to_balance_tf(t_dpo, t_cogs, period_days)
            inv_end = days_to_balance_tf(t_dio, t_cogs, period_days)

            def _slice_or_prev(ts_opt, prev_val):
                return s3(tm(ts_opt)[t]) if ts_opt is not None else prev_val
            # Other current assets: simulated delta only via fallback coefficients (no driver)
            delta_oca = s3(omega_oca_sales_tm[t]) * delta_sales + s3(omega_oca_opex_tm[t]) * opex
            t_other_cur = other_cur0 + delta_oca
            # Goodwill & other NCA now always derived from aggregate investing CF feature.
            if getattr(drivers, 'aggregate_invest', None) is not None:
                invest_feat = _slice_or_prev(drivers.aggregate_invest, tf.zeros_like(goodwill0))
            else:
                invest_feat = tf.zeros_like(goodwill0)
            t_change_goodwill = t_prem * tf.nn.relu(-invest_feat)
            t_goodwill = goodwill0 + t_change_goodwill

            # Other NCA: linear combo of capex and aggregate investing feature.
            if getattr(drivers, 'aggregate_invest', None) is not None:
                invest_other_feat = _slice_or_prev(drivers.aggregate_invest, tf.zeros_like(other_nca0))
            else:
                invest_other_feat = tf.zeros_like(other_nca0)

            t_change_onca = (
                t_b1 * tf.where(tf.math.is_nan(t_capex), tf.zeros_like(t_capex), t_capex)
                + t_b2 * invest_other_feat
            )
            t_other_nca = other_nca0 + t_change_onca
            # Flow-based current liabilities:
            # Accrued expenses: prior + change_in_accrued_expenses
            t_change_accr = _slice_or_prev(drivers.change_in_accrued_expenses, tf.zeros_like(accr_exp0))
            t_accr_exp = accr_exp0 + t_change_accr
            # Tax payable: prior + change_in_tax_payable (direct CF delta). If driver missing, approximate using taxes (no change).
            t_change_tax = _slice_or_prev(drivers.change_in_tax_payable, tf.zeros_like(tax_pay0))
            t_tax_pay = tax_pay0 + t_change_tax
            # Other non-current liabilities: simulated delta only via fallback coefficients (no driver)
            deferred_tax_feat2 = _slice_or_prev(getattr(drivers, 'deferred_income_tax', None), tf.zeros_like(other_ncl0)) if getattr(drivers, 'deferred_income_tax', None) is not None else tf.zeros_like(other_ncl0)
            other_non_cash_feat2 = _slice_or_prev(getattr(drivers, 'other_non_cash_items', None), tf.zeros_like(other_ncl0)) if getattr(drivers, 'other_non_cash_items', None) is not None else tf.zeros_like(other_ncl0)
            delta_oncl = s3(psi_oncl_deferred_tax_tm[t]) * deferred_tax_feat2 + s3(psi_oncl_other_nc_tm[t]) * other_non_cash_feat2
            t_other_ncl = other_ncl0 + delta_oncl
            # AOCI: single total delta driver. We no longer simulate components; instead
            # we roll forward the total balance using change_in_aoci when provided.
            if getattr(drivers, 'change_in_aoci', None) is not None:
                delta_aoci = _slice_or_prev(drivers.change_in_aoci, tf.zeros_like(aoci0))
            else:
                delta_aoci = tf.zeros_like(aoci0)
            t_aoci = aoci0 + delta_aoci
            # Minority interest: single-driver delta. If missing, assume zero delta (carry forward).
            delta_minority = _slice_or_prev(getattr(drivers, 'change_in_minority_interest', None), tf.zeros_like(minority0)) if getattr(drivers, 'change_in_minority_interest', None) is not None else tf.zeros_like(minority0)
            t_minority = minority0 + delta_minority
            # Lease obligations: prefer explicit deltas; else structured schedule per policy
            if getattr(drivers, 'change_in_current_capital_lease_obligation', None) is not None and getattr(drivers, 'change_in_long_term_capital_lease_obligation', None) is not None:
                delta_cur_lease = _slice_or_prev(drivers.change_in_current_capital_lease_obligation, tf.zeros_like(cur_lease0))
                delta_lt_lease = _slice_or_prev(drivers.change_in_long_term_capital_lease_obligation, tf.zeros_like(lt_lease0))
                t_cur_lease = cur_lease0 + delta_cur_lease
                t_lt_lease = lt_lease0 + delta_lt_lease
            else:
                # Features for additions: we now drive new leases only off capex and sales growth.
                capex_pos = tf.where(tf.math.is_nan(t_capex), tf.zeros_like(t_capex), tf.nn.relu(t_capex))
                d_sales_pos = tf.nn.relu(delta_sales)
                additions = (
                    t_lease_add_capex * capex_pos +
                    t_lease_add_sales * d_sales_pos
                )
                safe_term = tf.maximum(t_lease_avg_term, tf.constant(1e-6, dtype=tf.float32))
                reclass = tf.nn.relu(lt_lease0) / safe_term
                principal = tf.nn.relu(t_lease_princ_rate) * tf.nn.relu(cur_lease0)
                terminations = tf.nn.relu(t_lease_term_sales) * tf.nn.relu(-delta_sales)
                t_lt_lease = tf.nn.relu(lt_lease0 + additions - reclass - terminations)
                t_cur_lease = tf.nn.relu(cur_lease0 + reclass - principal - terminations)
            # Cash dividends paid driver (CF outflow magnitude). Fallback: assume cash paid equals prior liability if driver absent.
            t_cash_div_paid = _slice_or_prev(drivers.cash_dividends_paid, div_pay0)
            # Dividends payable roll-forward: liability accumulates declared minus paid.
            # div_pay_t = div_pay_prev + declared_dividends - cash_dividends_paid
            # This preserves identity: retained earnings reduced by declared; cash reduced only by paid portion; difference sits in liability.
            t_div_pay = div_pay0 + dividends - t_cash_div_paid
            # Deferred liabilities removed: folded into accrued expenses in prev state; no separate roll-forward
            # other_payables removed
            # Net common stock issuance driver
            t_net_stock_issuance = _slice_or_prev(getattr(drivers, 'net_common_stock_issuance', None), tf.zeros_like(paid_in_beg)) if getattr(drivers, 'net_common_stock_issuance', None) is not None else tf.zeros_like(paid_in_beg)
            # Capital stock simulated via gamma * max(0, net_common_stock_issuance)
            t_cap_stock = cap_stock0 + s3(gamma_cap_tm[t]) * tf.nn.relu(t_net_stock_issuance)

            # Deltas
            d_ar = ar_end - ar_beg; d_ap = ap_end - ap_beg; d_inv = inv_end - inv_beg
            d_other_cur = t_other_cur - other_cur0; d_accr_exp = t_change_accr; d_tax_pay = t_change_tax
            d_goodwill = t_goodwill - goodwill0; d_other_nca = t_other_nca - other_nca0; d_other_ncl = t_other_ncl - other_ncl0
            d_minority = t_minority - minority0
            d_cur_lease = t_cur_lease - cur_lease0; d_lt_lease = t_lt_lease - lt_lease0
            d_div_pay = t_div_pay - div_pay0; d_cap_stock = t_cap_stock - cap_stock0

            # Working capital change (positive consumes cash)
            # Working capital change: include liability increases (sources of cash) with negative sign
            # Working capital change: exclude dividends payable (financing distribution) and removed other_payables.
            nwc_change = d_ar + d_inv + d_other_cur - d_ap - d_accr_exp - d_tax_pay

            base_capex = tf.where(tf.math.is_nan(t_capex), depreciation, t_capex)
            capex = base_capex + d_goodwill + d_other_nca   # investing flows include non-current asset deltas

            cash_from_ops = net_income + depreciation - nwc_change
            cash_available = cash_beg + cash_from_ops
            cash_after_invest = cash_available - capex
            financing_external = d_other_ncl + d_minority + d_cur_lease + d_lt_lease + d_cap_stock + t_net_stock_issuance
            # Cash dividends paid actual driver value (already positive outflow magnitude)
            cash_dividends_paid = t_cash_div_paid
            after_div = cash_after_invest - cash_dividends_paid + financing_external

            min_cash = (t_cash_cov * (tf.nn.relu(t_cogs) + tf.nn.relu(opex) + tf.nn.relu(taxes))) if has_cov else (t_min_cash_ratio * tf.nn.relu(t_sales))
            deficit = 0.5 * tf.nn.relu(-after_div) # after_div < 0 indicates cash shortfall
            lt_raise = t_lt_share * deficit; st_draw_for_invest = deficit - lt_raise
            lt_end = lt_beg + lt_raise; st_mid = st_beg + st_draw_for_invest
            after_div = tf.where(deficit > 0.0, 0.0, after_div)
            shortfall = min_cash - after_div
            trigger = tf.nn.relu(shortfall - 0.1 * min_cash)  # buffer to avoid churning around min cash
            need = trigger
            short_term_end = st_mid + need
            cash_end = tf.where(need > 0.0, min_cash, after_div)
            excess = tf.nn.relu(cash_end - min_cash)
            sti_end = tf.nn.relu(sti_beg) + tf.where(need > 0.0, 0.0, excess)
            cash_end = cash_end - tf.where(need > 0.0, 0.0, excess)
            ppe_end = tf.nn.relu(ppe_beg + capex - depreciation)

            # Retained earnings reduce by declared dividends (dividends). Paid-in unaffected.
            retained_end = retained_beg + (net_income - dividends)
            d_lt = lt_end - lt_beg
            paid_in_end = paid_in_beg + k_pi * tf.nn.relu(d_lt)  # issuance increases paid-in; repurchase (negative) reduces
            equity_end = retained_end + paid_in_end  # capital_stock separate

            assets = cash_end + sti_end + ar_end + inv_end + t_other_cur + ppe_end + t_goodwill + t_other_nca
            liab_eq = (
                short_term_end + t_cur_lease + lt_end + t_lt_lease + ap_end +
                t_accr_exp + t_tax_pay + t_div_pay + t_other_ncl +
                (equity_end + t_cap_stock + t_aoci) + t_minority
            )
            gap = tf.abs(assets - liab_eq)

            if tf.executing_eagerly():
                tf.debugging.assert_all_finite(t_sales, "t_sales has NaN/Inf")
                tf.debugging.assert_all_finite(t_cogs, "t_cogs has NaN/Inf")
                if gross_margin_tm is not None:
                    tf.debugging.assert_all_finite(t_gm, "t_gm (gross margin) has NaN/Inf")
                tf.debugging.assert_all_finite(opex, "opex has NaN/Inf")
                tf.debugging.assert_all_finite(depreciation, "depreciation has NaN/Inf")
                tf.debugging.assert_all_finite(ebit, "ebit has NaN/Inf")
                tf.debugging.assert_all_finite(ebt, "ebt has NaN/Inf")
                tf.debugging.assert_all_finite(taxes, "taxes has NaN/Inf")
                tf.debugging.assert_all_finite(net_income, "net_income has NaN/Inf")
                tf.debugging.assert_all_finite(depreciation, "depreciation has NaN/Inf")
                tf.debugging.assert_all_finite(nwc_change, "nwc_change has NaN/Inf")
                tf.debugging.assert_all_finite(cash_from_ops, "cash_from_ops has NaN/Inf")
                tf.debugging.assert_all_finite(base_capex, "base_capex has NaN/Inf")
                tf.debugging.assert_all_finite(capex, "capex has NaN/Inf")
                tf.debugging.assert_all_finite(financing_external, "financing_external has NaN/Inf")

            if self.collect_diagnostics:
                diag_rows.append(tf.concat([
                    cash_end, sti_end, ar_end, inv_end, t_other_cur, ppe_end, t_goodwill, t_other_nca,
                    short_term_end, t_cur_lease, lt_end, t_lt_lease, ap_end, t_accr_exp, t_tax_pay, t_div_pay, t_other_ncl,
                    equity_end, t_cap_stock, t_aoci, t_minority,
                    assets, liab_eq, gap, cash_from_ops, capex, financing_external, nwc_change,
                    ebit, ebt, taxes, net_income, depreciation, base_capex
                ], axis=-1))
            if self.hard_identity_check and tf.executing_eagerly():
                tf.debugging.assert_less_equal(gap, self.identity_tol, message="Accounting identity violated in WMT layer")

            vals = [
                t_sales, t_cogs, (t_sales - t_cogs), opex, ebit,
                interest_income, interest_expense, ebt, taxes, net_income,
                capex, depreciation, nwc_change, cash_end, sti_end,
                short_term_end, lt_end, ar_end, ap_end, inv_end, ppe_end,
                equity_end, dividends, retained_end, paid_in_end,
                t_other_cur, t_goodwill, t_other_nca, t_accr_exp, t_tax_pay, t_other_ncl, t_aoci, t_minority,
                t_cur_lease, t_lt_lease, t_div_pay, t_cap_stock
            ]
            for i, v in enumerate(vals):
                outs[i].append(v)

            # advance carry
            cash_beg, sti_beg, st_beg, lt_beg = cash_end, sti_end, short_term_end, lt_end
            ar_beg, ap_beg, inv_beg, ppe_beg = ar_end, ap_end, inv_end, ppe_end
            equity_beg, retained_beg, paid_in_beg = equity_end, retained_end, paid_in_end
            other_cur0, goodwill0, other_nca0 = t_other_cur, t_goodwill, t_other_nca
            accr_exp0, tax_pay0, other_ncl0 = t_accr_exp, t_tax_pay, t_other_ncl
            aoci0 = t_aoci
            minority0 = t_minority
            cur_lease0, lt_lease0 = t_cur_lease, t_lt_lease
            div_pay0, cap_stock0 = t_div_pay, t_cap_stock

        # Stack and sanitize NaNs
        stacked = [tf.squeeze(tf.stack(seq, axis=0), axis=-1) for seq in outs]
        stacked = [tf.where(tf.math.is_nan(x), tf.zeros_like(x), x) for x in stacked]
        (
            o_sales, o_cogs, o_gross, o_opex, o_ebit,
            o_int_inc, o_int_exp, o_ebt, o_taxes, o_net_inc,
            o_capex, o_depr, o_wcchg, o_cash, o_sti,
            o_st_debt, o_lt_debt, o_ar, o_ap, o_inv, o_ppe,
            o_equity, o_div, o_retained, o_paid_in,
            o_other_cur, o_goodwill, o_other_nca, o_accr_exp, o_tax_pay, o_other_ncl, o_aoci, o_minority,
            o_cur_lease, o_lt_lease, o_div_pay, o_cap_stock
        ) = stacked

        bt = lambda x: tf.transpose(x, [1, 0, 2])
        stm = StatementsWMT(
            sales=bt(o_sales), cogs=bt(o_cogs), gross_profit=bt(o_gross), opex=bt(o_opex), ebit=bt(o_ebit),
            interest_income=bt(o_int_inc), interest_expense=bt(o_int_exp), ebt=bt(o_ebt), taxes=bt(o_taxes), net_income=bt(o_net_inc),
            capex=bt(o_capex), depreciation=bt(o_depr), wc_change=bt(o_wcchg),
            cash=bt(o_cash), st_investments=bt(o_sti), st_debt=bt(o_st_debt), lt_debt=bt(o_lt_debt),
            ar=bt(o_ar), ap=bt(o_ap), inventory=bt(o_inv), net_ppe=bt(o_ppe), equity=bt(o_equity),
            dividends=bt(o_div), retained_earnings=bt(o_retained), paid_in_capital=bt(o_paid_in),
            other_current_assets=bt(o_other_cur), goodwill_intangibles=bt(o_goodwill), other_non_current_assets=bt(o_other_nca),
            accrued_expenses=bt(o_accr_exp), tax_payable=bt(o_tax_pay), other_non_current_liabilities=bt(o_other_ncl),
            aoci=bt(o_aoci), minority_interest=bt(o_minority),
            current_capital_lease_obligation=bt(o_cur_lease), long_term_capital_lease_obligation=bt(o_lt_lease),
            dividends_payable=bt(o_div_pay), capital_stock=bt(o_cap_stock)
        )

        if self.hard_identity_check and tf.executing_eagerly():
            gap_chk = tf.reduce_max(tf.abs(stm.assets - stm.liab_plus_equity))
            tf.debugging.assert_less_equal(gap_chk, self.identity_tol, message="Accounting identity violated in WMT layer")

        if self.collect_diagnostics:
            diag_tensor = tf.transpose(tf.stack(diag_rows, axis=0), [1,0,2,3])
            diag_tensor = tf.squeeze(diag_tensor, axis=2)  # [B,T,F]
            self.last_diagnostics = {
                "columns": [
                    "cash","sti","ar","inv","other_cur","net_ppe","goodwill","other_nca",
                    "st_debt","cur_lease","lt_debt","lt_lease","ap","accrued_exp","tax_pay","div_pay","other_ncl",
                    "equity","cap_stock","aoci","minority",
                    "assets_total","liab_eq_total","gap","cfo","cfi","fin_ext","nwc_change",
                    "ebit","ebt","taxes","net_income","depr","capex_base"
                ],
                "tensor": diag_tensor
            }
        return stm

    def run(self, drivers: DriversWMT, policies: PoliciesWMT, prev: PrevStateWMT) -> List[StatementsWMT]:
        stm = self.call(drivers, policies, prev, training=False)
        Tn = tf.shape(stm.sales)[1]
        out: List[StatementsWMT] = []
        for i in range(int(Tn.numpy())):
            idx = slice(i, i+1)
            out.append(StatementsWMT(
                sales=stm.sales[:, idx, :], cogs=stm.cogs[:, idx, :], gross_profit=stm.gross_profit[:, idx, :],
                opex=stm.opex[:, idx, :], ebit=stm.ebit[:, idx, :], interest_income=stm.interest_income[:, idx, :],
                interest_expense=stm.interest_expense[:, idx, :], ebt=stm.ebt[:, idx, :], taxes=stm.taxes[:, idx, :], net_income=stm.net_income[:, idx, :],
                capex=stm.capex[:, idx, :], depreciation=stm.depreciation[:, idx, :], wc_change=stm.wc_change[:, idx, :],
                cash=stm.cash[:, idx, :], st_investments=stm.st_investments[:, idx, :], st_debt=stm.st_debt[:, idx, :], lt_debt=stm.lt_debt[:, idx, :],
                ar=stm.ar[:, idx, :], ap=stm.ap[:, idx, :], inventory=stm.inventory[:, idx, :], net_ppe=stm.net_ppe[:, idx, :], equity=stm.equity[:, idx, :],
                dividends=stm.dividends[:, idx, :], retained_earnings=stm.retained_earnings[:, idx, :], paid_in_capital=stm.paid_in_capital[:, idx, :],
                other_current_assets=stm.other_current_assets[:, idx, :], goodwill_intangibles=stm.goodwill_intangibles[:, idx, :], other_non_current_assets=stm.other_non_current_assets[:, idx, :],
                accrued_expenses=stm.accrued_expenses[:, idx, :], tax_payable=stm.tax_payable[:, idx, :], other_non_current_liabilities=stm.other_non_current_liabilities[:, idx, :],
                aoci=stm.aoci[:, idx, :], minority_interest=stm.minority_interest[:, idx, :],
                current_capital_lease_obligation=stm.current_capital_lease_obligation[:, idx, :], long_term_capital_lease_obligation=stm.long_term_capital_lease_obligation[:, idx, :],
                dividends_payable=stm.dividends_payable[:, idx, :], capital_stock=stm.capital_stock[:, idx, :]
            ))
        return out
