import numpy as np
import tensorflow as tf

from wmt_bs_forecaster.losses_wmt import wmt_fit_loss
from wmt_bs_forecaster.types_wmt import StatementsWMT


def _series(vals):
    arr = np.array(vals, dtype=np.float32).reshape(1, -1, 1)
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def _blank_statements(T: int = 3) -> StatementsWMT:
    z = lambda: _series([0.0] * T)
    return StatementsWMT(
        sales=z(), cogs=z(), gross_profit=z(), opex=z(), ebit=z(),
        interest_income=z(), interest_expense=z(), ebt=z(), taxes=z(), net_income=z(),
        capex=z(), depreciation=z(), wc_change=z(),
        cash=z(), st_investments=z(), st_debt=z(), lt_debt=z(),
        ar=z(), ap=z(), inventory=z(), net_ppe=z(), equity=z(),
        dividends=z(), retained_earnings=z(), paid_in_capital=z(),
        other_current_assets=z(), goodwill_intangibles=z(), other_non_current_assets=z(),
        accrued_expenses=z(), tax_payable=z(), other_non_current_liabilities=z(),
        aoci=z(), minority_interest=z(),
        current_capital_lease_obligation=z(), long_term_capital_lease_obligation=z(),
        dividends_payable=z(), capital_stock=z(),
    )


def test_wmt_fit_loss_zero_when_perfect_match(tmp_path):
    # Build a tiny synthetic history where model == target on all used lines.
    T = 3
    stm = _blank_statements(T)

    # Create a matching pandas DataFrames for balance sheet and financials.
    import pandas as pd

    cols = ["t0", "t1", "t2"]
    zeros = np.zeros(T, dtype=np.float32)

    bal = pd.DataFrame(
        {
            "Cash And Cash Equivalents": zeros,
            "Accounts Receivable": zeros,
            "Accounts Payable": zeros,
            "Inventory": zeros,
            "Other Current Assets": zeros,
            "Net PPE": zeros,
            "Stockholders Equity": zeros,
            "Gains Losses Not Affecting Retained Earnings": zeros,
            "Goodwill": zeros,
            "Other Non Current Assets": zeros,
            "Other Non Current Liabilities": zeros,
            "Current Capital Lease Obligation": zeros,
            "Long Term Capital Lease Obligation": zeros,
            "Current Debt": zeros,
            "Long Term Debt": zeros,
            "Total Tax Payable": zeros,
        },
        index=cols,
    ).T

    fin = pd.DataFrame(
        {
            "Total Revenue": zeros,
            "Gross Profit": zeros,
            "Operating Expense": zeros,
            "Reconciled Depreciation": zeros,
            "Net Income": zeros,
        },
        index=cols,
    ).T

    # Identity gap is also zero because all BS components are zero.
    targets = {
        "cash": tf.convert_to_tensor(
            bal.loc["Cash And Cash Equivalents", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "ar": tf.convert_to_tensor(
            bal.loc["Accounts Receivable", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "ap": tf.convert_to_tensor(
            bal.loc["Accounts Payable", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "inventory": tf.convert_to_tensor(
            bal.loc["Inventory", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "other_current_assets": tf.convert_to_tensor(
            bal.loc["Other Current Assets", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "net_ppe": tf.convert_to_tensor(
            bal.loc["Net PPE", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "equity": tf.convert_to_tensor(
            bal.loc["Stockholders Equity", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "aoci": tf.convert_to_tensor(
            bal.loc["Gains Losses Not Affecting Retained Earnings", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "goodwill_intangibles": tf.convert_to_tensor(
            bal.loc["Goodwill", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "other_non_current_assets": tf.convert_to_tensor(
            bal.loc["Other Non Current Assets", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "other_non_current_liabilities": tf.convert_to_tensor(
            bal.loc["Other Non Current Liabilities", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "current_capital_lease_obligation": tf.convert_to_tensor(
            bal.loc["Current Capital Lease Obligation", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "long_term_capital_lease_obligation": tf.convert_to_tensor(
            bal.loc["Long Term Capital Lease Obligation", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "st_debt": tf.convert_to_tensor(
            bal.loc["Current Debt", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "lt_debt": tf.convert_to_tensor(
            bal.loc["Long Term Debt", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "tax_payable": tf.convert_to_tensor(
            bal.loc["Total Tax Payable", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "sales": tf.convert_to_tensor(
            fin.loc["Total Revenue", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "gross_profit": tf.convert_to_tensor(
            fin.loc["Gross Profit", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "opex": tf.convert_to_tensor(
            fin.loc["Operating Expense", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "depreciation": tf.convert_to_tensor(
            fin.loc["Reconciled Depreciation", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "net_income": tf.convert_to_tensor(
            fin.loc["Net Income", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
    }

    losses = wmt_fit_loss(stm, targets, include_identity=True, include_retained=False)
    total = float(losses["total"].numpy())
    assert total == 0.0


def test_wmt_fit_loss_increases_with_larger_mismatch():
    """If we move the model further from the target, loss should increase."""
    T = 3
    stm = _blank_statements(T)

    import pandas as pd

    cols = ["t0", "t1", "t2"]
    zeros = np.zeros(T, dtype=np.float32)

    # Target balance sheet and financials: all zeros for cash and sales.
    bal = pd.DataFrame(
        {
            "Cash And Cash Equivalents": zeros,
        },
        index=cols,
    ).T

    fin = pd.DataFrame(
        {
            "Total Revenue": zeros,
        },
        index=cols,
    ).T

    targets = {
        "cash": tf.convert_to_tensor(
            bal.loc["Cash And Cash Equivalents", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
        "sales": tf.convert_to_tensor(
            fin.loc["Total Revenue", cols].to_numpy(dtype=np.float32).reshape(1, -1, 1)
        ),
    }

    # Case 1: model == target (cash and sales zero) -> smaller loss.
    loss0 = wmt_fit_loss(stm, targets)["total"]

    # Case 2: move model away from target on cash and sales.
    stm_perturbed = _blank_statements(T)
    stm_perturbed.cash = _series([1.0, 1.0, 1.0])
    stm_perturbed.sales = _series([1.0, 1.0, 1.0])

    loss1 = wmt_fit_loss(stm_perturbed, targets)["total"]

    assert float(loss1.numpy()) > float(loss0.numpy())
