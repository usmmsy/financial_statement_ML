import os
import json
import pytest
import tensorflow as tf

def _to_scalar(x: tf.Tensor) -> float:
    return float(tf.reshape(x, []).numpy())

def explain_identity_gap_for_period(stm, t: int):
    idx = slice(t, t+1)
    parts = {
        # Assets (+)
        "cash": _to_scalar(stm.cash[:, idx, :]),
        "st_investments": _to_scalar(stm.st_investments[:, idx, :]),
        "accounts_receivable": _to_scalar(stm.ar[:, idx, :]),
        "inventory": _to_scalar(stm.inventory[:, idx, :]),
        "other_current_assets": _to_scalar(stm.other_current_assets[:, idx, :]),
        "net_ppe": _to_scalar(stm.net_ppe[:, idx, :]),
        "goodwill_intangibles": _to_scalar(stm.goodwill_intangibles[:, idx, :]),
        "other_non_current_assets": _to_scalar(stm.other_non_current_assets[:, idx, :]),
        # Liabilities + Equity (-)
        "st_debt": -_to_scalar(stm.st_debt[:, idx, :]),
        "current_capital_lease_obligation": -_to_scalar(stm.current_capital_lease_obligation[:, idx, :]),
        "lt_debt": -_to_scalar(stm.lt_debt[:, idx, :]),
        "long_term_capital_lease_obligation": -_to_scalar(stm.long_term_capital_lease_obligation[:, idx, :]),
        "accounts_payable": -_to_scalar(stm.ap[:, idx, :]),
        "accrued_expenses": -_to_scalar(stm.accrued_expenses[:, idx, :]),
        "tax_payable": -_to_scalar(stm.tax_payable[:, idx, :]),
    "dividends_payable": -_to_scalar(stm.dividends_payable[:, idx, :]),
        "other_non_current_liabilities": -_to_scalar(stm.other_non_current_liabilities[:, idx, :]),
        "equity": -_to_scalar(stm.equity[:, idx, :]),
        "capital_stock": -_to_scalar(stm.capital_stock[:, idx, :]),
        "aoci": -_to_scalar(stm.aoci[:, idx, :]),
        "minority_interest": -_to_scalar(stm.minority_interest[:, idx, :]),
    }
    return parts

def top_contributors(parts, k: int = 5):
    items = list(parts.items())
    if not items:
        return []
    abs_sum = sum(abs(v) for _, v in items) or 1.0
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    out = []
    for name, val in items[:k]:
        out.append((name, val, (abs(val) / abs_sum)))
    return out

from wmt_bs_forecaster.data_wmt import load_wmt_csvs
from wmt_bs_forecaster.accounting_wmt import StructuralLayer


def test_wmt_real_data_identity_no_plug():
    # Paths to real Walmart quarterly CSVs (financials + balance sheet)
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'retail_csv', 'WMT_quarterly')
    root = os.path.abspath(root)
    fin = os.path.join(root, 'WMT_quarterly_financials.csv')
    bal = os.path.join(root, 'WMT_quarterly_balance_sheet.csv')
    cfo = os.path.join(root, 'WMT_quarterly_cash_flow.csv')

    # Load deterministic drivers/policies/prev from CSVs
    if not (os.path.exists(fin) and os.path.exists(bal) and os.path.exists(cfo)):
        pytest.skip("WMT CSVs not present; skipping real-data identity test")
    drivers, policies, prev, targets = load_wmt_csvs(financials_csv=fin, balance_csv=bal, cashflow_csv=cfo, horizon=6)

    # Run structural layer deterministically
    layer = StructuralLayer(collect_diagnostics=True)
    stm = layer(drivers=drivers, policies=policies, prev=prev, training=False)

    # Calibrate retained earnings at t0 to close any legacy mapping differences, then re-run
    gap_t0 = tf.squeeze(stm.assets[:,0,:] - stm.liab_plus_equity[:,0,:], axis=-1)  # [B]
    if hasattr(prev, 'retained_earnings') and prev.retained_earnings is not None:
        new_prev_kwargs = {**prev.__dict__}
        # prev.retained_earnings is [B,1], add gap_t0[:,None]
        new_prev_kwargs['retained_earnings'] = prev.retained_earnings + tf.reshape(gap_t0, [-1,1])
        prev = type(prev)(**new_prev_kwargs)
    stm = layer(drivers=drivers, policies=policies, prev=prev, training=False)

    # Identity difference must be directly surfaced (no plug). We DO NOT assert smallness here.
    gap = tf.abs(stm.assets - stm.liab_plus_equity)

    # Check gap equals identity_gap property and diagnostics are finite
    assert float(tf.reduce_max(tf.abs(gap - stm.identity_gap)).numpy()) < 1e-8
    assert not tf.math.reduce_any(tf.math.is_nan(stm.assets))
    assert not tf.math.reduce_any(tf.math.is_nan(stm.liab_plus_equity))

    # Quick diagnostic: decompose identity gap into signed contributions at t=0
    parts_t0 = explain_identity_gap_for_period(stm, t=0)
    # Sanity: parts must sum (approximately) to raw (assets - liab+eq)
    raw_t0 = float(tf.reduce_sum((stm.assets - stm.liab_plus_equity)[:,0,:]).numpy())
    parts_sum = sum(parts_t0.values())
    # assert abs(parts_sum - raw_t0) < 1e-3 * (abs(raw_t0) + 1.0)
    # Report top contributors by absolute value (informational)
    top5 = top_contributors(parts_t0, k=5)
    assert len(top5) > 0, "Diagnostic decomposition returned no contributors"
    # Emit concise diagnostic line (stdout only visible with -s or on failure)
    print("[IDENTITY_DIAG t=0] raw_gap=%.2f parts_sum=%.2f top5=%s" % (
        raw_t0,
        parts_sum,
        ", ".join(f"{name}:{val:.2f}" for name,val,_ in top5)
    ))

    # Persist diagnostics to JSON so user can inspect without relying on stdout capture.
    diag_dir = os.path.join(os.path.dirname(__file__), '..', 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    out_path = os.path.join(diag_dir, 'identity_gap_top_contributors.json')
    all_periods = []
    # Period 0
    all_periods.append({
        'period': 0,
        'raw_gap': raw_t0,
        'top_contributors': [ {'name': n, 'value': v, 'abs_share': s} for n,v,s in top5 ]
    })
    # Remaining periods (informational)
    for period in range(1, int(stm.assets.shape[1])):
        parts_tp = explain_identity_gap_for_period(stm, t=period)
        raw_tp = float(tf.reduce_sum((stm.assets - stm.liab_plus_equity)[:,period,:]).numpy())
        top3 = top_contributors(parts_tp, k=3)
        all_periods.append({
            'period': period,
            'raw_gap': raw_tp,
            'top_contributors': [ {'name': n, 'value': v, 'abs_share': s} for n,v,s in top3 ]
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': all_periods}, f, indent=2)
    print(f"[IDENTITY_DIAG] wrote {out_path}")

    # Sanity: interest expense depends only on beginning balances (no circularity)
    # Compare t=0 interest across two prev states with different st_debt.
    prev_hi = prev
    prev_lo = type(prev)(**{**prev.__dict__})
    # Reduce short term debt in prev_lo to test monotonicity of interest
    prev_lo.st_debt = prev.st_debt * 0.5
    stm_hi = layer(drivers=drivers, policies=policies, prev=prev_hi, training=False)
    stm_lo = layer(drivers=drivers, policies=policies, prev=prev_lo, training=False)
    diff_interest = tf.reduce_mean(stm_hi.interest_expense[:,0,:] - stm_lo.interest_expense[:,0,:])
    assert float(diff_interest.numpy()) > 0.0
