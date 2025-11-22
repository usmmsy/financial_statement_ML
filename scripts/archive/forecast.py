from __future__ import annotations

"""
Deterministic forecast demo using observed/derived drivers from yfinance data.

What this does:
- Loads quarterly data via YFinanceLoader
- Derives per-period drivers from observed statements:
  * price := 1.0, volume := sales (so sales = price * volume)
  * dso/dpo/dio from AR/AP/Inventory vs Sales/COGS
  * capex from features: capex_intensity = -capex/sales => capex = relu(-capex_intensity * sales)
  * stlt_split from positive deltas of short vs long-term debt
- Runs StructuralLayer sequentially from boundary balances at t=0 over t=1..T-1
- Reports accounting identity gap and MAE fit vs selected targets

Notes:
- Policies are the defaults broadcast by the loader (can be swapped for scenarios)
- This is purely deterministic (no DriverHead) to make wiring visible end-to-end
"""

from typing import Dict, List, Tuple
import argparse

import tensorflow as tf

from balance_sheet_forecaster.data import YFinanceLoader, DummyData, _FEATURE_NAMES
from balance_sheet_forecaster.types import Drivers, PrevState, Policies, Statements
from balance_sheet_forecaster.accounting import StructuralLayer
from balance_sheet_forecaster.model import BalanceSheetForecastModel
from balance_sheet_forecaster.rollout import rollout as rollout_seq


def _safe_div(n: tf.Tensor, d: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
	d_abs = tf.maximum(tf.abs(d), eps)
	return n / d_abs


def _derive_observed_drivers(
	features: tf.Tensor,          # [B,T,F]
	targets: Dict[str, tf.Tensor] # dict[str->[B,T,1]]
) -> Drivers:
	"""Build a Drivers object from observed statements and engineered features.
	
	This is for purely deterministic runs using observed data.

	Conventions:
	- price := 1.0; volume := sales so that sales = price * volume
	- dso_days := AR/Sales * 365; dpo_days := AP/COGS * 365; dio_days := Inventory/COGS * 365
	- capex_out := relu(-capex_intensity * sales)
	- stlt_split := positive LT debt delta / positive total debt delta (fallback 0.5 when denom ~ 0)
	"""
	B, T, F = features.shape

	# Sales and COGS from targets
	sales = tf.convert_to_tensor(targets["sales"], tf.float32)       # [B,T,1]
	cogs  = tf.convert_to_tensor(targets["cogs"], tf.float32)        # [B,T,1]

	# Working-capital balances
	ar = tf.convert_to_tensor(targets["ar"], tf.float32)
	ap = tf.convert_to_tensor(targets["ap"], tf.float32)
	inv = tf.convert_to_tensor(targets["inventory"], tf.float32)

	# Days on hand
	dso_days = _safe_div(ar, tf.maximum(sales, 1e-6)) * 365.0
	dpo_days = _safe_div(ap, tf.maximum(cogs, 1e-6)) * 365.0
	dio_days = _safe_div(inv, tf.maximum(cogs, 1e-6)) * 365.0

	# price and volume (degenerate split)
	price  = tf.ones_like(sales)
	volume = sales

	# Capex from features: capex_intensity = -capex/sales
	capex_idx = _FEATURE_NAMES.index("capex_intensity")
	capex_intensity = features[..., capex_idx:capex_idx+1]  # [B,T,1]
	capex = tf.nn.relu(-capex_intensity * sales)             # positive outflow magnitude

	# Debt split from deltas
	st_debt = tf.convert_to_tensor(targets["st_debt"], tf.float32)
	lt_debt = tf.convert_to_tensor(targets["lt_debt"], tf.float32)
	# deltas along time (pad first step with zeros)
	dst = tf.concat([tf.zeros_like(st_debt[:, :1, :]), st_debt[:, 1:, :] - st_debt[:, :-1, :]], axis=1)
	dlt = tf.concat([tf.zeros_like(lt_debt[:, :1, :]), lt_debt[:, 1:, :] - lt_debt[:, :-1, :]], axis=1)
	pos_dst = tf.nn.relu(dst)
	pos_dlt = tf.nn.relu(dlt)
	denom = pos_dst + pos_dlt + 1e-6
	stlt_split = pos_dlt / denom  # in [0,1]

	return Drivers(
		price=price,
		volume=volume,
		dso_days=dso_days,
		dpo_days=dpo_days,
		dio_days=dio_days,
		capex=capex,
		stlt_split=stlt_split,
	)


def _prev_from_targets_at(
	targets: Dict[str, tf.Tensor],
	t: int,
) -> PrevState:
	"""Build PrevState from balance sheet lines at a specific time index t."""
	sl = slice(t, t+1)
	def pick(name: str) -> tf.Tensor:
		return tf.convert_to_tensor(targets[name][:, sl, :], tf.float32)  # [B,1,1]
	return PrevState(
		cash=pick("cash"),
		st_investments=pick("st_investments"),
		st_debt=pick("st_debt"),
		lt_debt=pick("lt_debt"),
		ar=pick("ar"),
		ap=pick("ap"),
		inventory=pick("inventory"),
		nfa=pick("nfa"),
		equity=pick("equity"),
	)

def _mae(x: tf.Tensor) -> tf.Tensor:
	return tf.reduce_mean(tf.abs(x))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Deterministic forecast using yfinance fundamentals.")
	parser.add_argument(
		"--tickers",
		nargs="+",
		default=["AAPL", "MSFT", "GOOGL"],
		help="List of tickers to load (space-separated)",
	)
	parser.add_argument(
		"--horizon",
		type=int,
		default=8,
		help="Number of most recent quarters required per ticker",
	)
	return parser.parse_args()


def main():
	# 1) Parse CLI and load data via yfinance
	args = parse_args()
	print(f"[config] tickers={args.tickers} horizon={args.horizon}")
	try:
		loader = YFinanceLoader(tickers=args.tickers, horizon_quarters=args.horizon)
		features, policies, prev, targets, tickers = loader.all()
		B, T, F = features.shape
		print(f"[yfinance] Loaded B={B}, T={T}, F={F} for tickers: {tickers}")
	except Exception as e_yf:
		msg = (
			"[error] Failed to load real data via yfinance.\n"
			f"Reason: {e_yf}\n\n"
			"Please check: \n"
			"  - Internet connectivity and firewall/proxy settings\n"
			"  - yfinance is installed and up to date\n"
			"  - Tickers exist and have sufficient quarterly history\n"
			"  - Temporary rate limiting from Yahoo (retry later)\n"
		)
		print(msg)
		raise SystemExit(1)

	# 2) Derive observed drivers
	drivers_obs = _derive_observed_drivers(features, targets)

	# 3) Prepare boundary prev at t=0 and slice forecast window t=1..T-1
	prev_last = _prev_from_targets_at(targets, t=0)
	features_f = features[:, 1:, :]
	# Slice policies across the same window using dataclass slicing
	policies_f = policies[:, 1:, :]
	# Slice drivers for forecast window
	def sl1(x: tf.Tensor) -> tf.Tensor:
		return x[:, 1:, :]
	drivers_f = Drivers(
		price=sl1(drivers_obs.price),
		volume=sl1(drivers_obs.volume),
		dso_days=sl1(drivers_obs.dso_days),
		dpo_days=sl1(drivers_obs.dpo_days),
		dio_days=sl1(drivers_obs.dio_days),
		capex=sl1(drivers_obs.capex),
		stlt_split=sl1(drivers_obs.stlt_split),
	)

	# 4) Rollout sequential accounting with drivers override
	model = BalanceSheetForecastModel(hidden=64)
	stm_pred, _ = rollout_seq(
		model,
		features=features_f,
		policies_roll=policies_f,
		prev_last=prev_last,
		drivers_override=drivers_f,
		training=False,
		return_drivers=False,
	)

	# 5) Compare to observed targets on overlapping horizon (t=1..T-1)
	targs_slice = {k: v[:, 1:, :] for k, v in targets.items()}

	keys = [
		"sales", "cogs", "opex", "net_income",
		"cash", "ar", "ap", "inventory", "st_investments",
		"st_debt", "lt_debt", "nfa", "equity",
	]

	print("\nDeterministic run metrics (t=1..T-1):")
	gaps = tf.reduce_mean(tf.abs(stm_pred.assets - stm_pred.liab_plus_equity))
	print(f"  Avg identity gap (|Assets - L+E|): {float(gaps.numpy()):.6f}")

	for k in keys:
		if k in targs_slice:
			pred_line = getattr(stm_pred, k)
			true_line = targs_slice[k]
			m = _mae(pred_line - true_line)
			print(f"  MAE[{k:>12}]: {float(m.numpy()):.4f}")


if __name__ == "__main__":
	main()

