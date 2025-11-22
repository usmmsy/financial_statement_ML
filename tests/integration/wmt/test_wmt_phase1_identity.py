import os
import pytest
import tensorflow as tf
from wmt_bs_forecaster.data_wmt import load_wmt_csvs
from wmt_bs_forecaster.accounting_wmt import StructuralLayer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'retail_csv', 'WMT_quarterly')
FIN = os.path.join(DATA_DIR, 'WMT_quarterly_financials.csv')
BS = os.path.join(DATA_DIR, 'WMT_quarterly_balance_sheet.csv')
CF = os.path.join(DATA_DIR, 'WMT_quarterly_cash_flow.csv')

def test_core_identity_and_retained_roll():
    if not (os.path.exists(FIN) and os.path.exists(BS) and os.path.exists(CF)):
        pytest.skip("WMT CSVs not present; skipping real-data identity test")
    drivers, policies, prev, targets = load_wmt_csvs(FIN, BS, CF, horizon=5)
    layer = StructuralLayer(hard_identity_check=False)
    stm = layer.call(drivers, policies, prev)

    # Expanded identity over full balance sheet (preferred for complete modeling)
    full_gap = tf.reduce_max(tf.abs(stm.assets - stm.liab_plus_equity)).numpy()
    assert full_gap < 1e-5, f"Expanded identity gap too large: {full_gap}"

    retained = tf.squeeze(stm.retained_earnings).numpy()
    net_income = tf.squeeze(stm.net_income).numpy()
    dividends = tf.squeeze(stm.dividends).numpy()
    for t in range(1, len(retained)):
        roll_gap = retained[t] - retained[t-1] - (net_income[t] - dividends[t])
        assert abs(roll_gap) < 1e-5, f"Retained earnings roll mismatch at t={t}: {roll_gap}"

    # Equity composition excludes minority interest (stored separately)
    equity_calc = tf.squeeze(stm.retained_earnings + stm.paid_in_capital).numpy()
    equity_reported = tf.squeeze(stm.equity).numpy()
    eq_gap = (equity_calc - equity_reported)
    assert (abs(eq_gap) < 1e-5).all(), f"Equity decomposition mismatch: {eq_gap}"

    # Diagnostic mode optional
    layer_diag = StructuralLayer(collect_diagnostics=True)
    _ = layer_diag.call(drivers, policies, prev)
    assert layer_diag.last_diagnostics is not None
    cols = layer_diag.last_diagnostics['columns']
    tensor = layer_diag.last_diagnostics['tensor']
    assert tensor.shape[-1] == len(cols)
    assert 'gap' in cols
