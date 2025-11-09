# Balance Sheet Forecaster (TensorFlow) — Skeleton

> Deterministic accounting engine + driver head to forecast financial statements with strict accounting identity (Assets = Liabilities + Equity), no plugs, and no circularity.

---

## ✨ Features
- **Structural layer** (`StructuralLayer`) that enforces clean treasury mechanics and identity.
- **DriverHead** GRU predicting behavioral levers (price, volume, DSO/DPO/DIO, capex, ST/LT mix).
- **Typed containers** for drivers, policies, previous state, statements.
- **Yahoo Finance loader** (optional; avoid in CI) plus a fully **deterministic DummyData** loader.
- **Unit tests** covering accounting identity and key policy behaviors.

---

## 📦 Repo layout
```
.
├─ src/balance_sheet_forecaster/
│  ├─ __init__.py              
│  ├─ accounting.py            # StructuralLayer
│  ├─ drivers.py               # DriverHead (GRU)
│  ├─ data.py                  # Yahoo loader + DummyData
│  ├─ config.py                # JSON/YAML config helper
│  ├─ types.py                 # Typed containers (dataclasses / NamedTuples)
│  └─ ...
├─ tests/
│  ├─ unit/
│  │  ├─ test_accounting.py
│  │  ├─ test_cash_policy_regression.py
│  │  ├─ test_config.py
│  │  ├─ test_data_offline.py
│  │  ├─ test_drivers.py
│  │  ├─ test_losses.py
│  │  ├─ test_model.py
│  │  ├─ test_rollout_unit.py
│  │  ├─ test_training.py
│  │  ├─ test_types.py
│  │  └─ test_utils_logging.py
│  └─ integration/
│     └─ 
├─ envs/
│  ├─ environment-macos-intel.yml
│  └─ environment-windows.yml
└─ README.md (this file)
```

---

## 🛠️ Setup

### 1) Create the environment
Pick the file for your platform and create the `bsforecast` env.

**macOS (Intel)**
```bash
conda env create -f envs/environment-macos-intel.yml
conda activate bsforecast
```

**Windows (PowerShell)**
```powershell
conda env create -f envs\environment-windows.yml
conda activate bsforecast
```

> Upgrade/downgrade later:
> ```bash
> conda env update -f envs/environment-macos-intel.yml --prune
> ```

### 2) Dev path
From repo root, point Python at `src/`:
- **bash/zsh (macOS/Linux):** `export PYTHONPATH=src`
- **PowerShell (Windows):** `$env:PYTHONPATH="src"`

(You can add this to your shell profile for convenience.)

---

## ✅ Run tests
From repo root:

**macOS/Linux**
```bash
export PYTHONPATH=src
pytest -q tests/unit/test_accounting.py
```

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH="src"
pytest -q tests/unit\test_accounting.py
```

You should see `8 passed` when everything is wired correctly.

---

## 🚀 Quickstart (dummy data)
Example sketch for integration (once `model.py` / `rollout.py` are filled):
```python
from balance_sheet_forecaster.drivers import DriverHead
from balance_sheet_forecaster.accounting import StructuralLayer
from balance_sheet_forecaster.types import Policies, PrevState
from balance_sheet_forecaster.data import DummyData

B, T = 2, 8
dummy = DummyData(B=B, T=T)
features = dummy.features()          # [B, T, F]
policies = dummy.policies()          # fields [B, T, 1]
prev = dummy.prev()                  # fields [B, 1]

head = DriverHead(hidden=64)
struct = StructuralLayer()

drivers = head(features, training=False)
stm = struct(drivers=drivers, policies=policies, prev=prev, training=False)
print('Assets ≈ L+E (mean):', float((stm.assets - stm.liab_plus_equity).numpy().mean()))
```

---

## ⚠️ Troubleshooting
- **TensorFlow vs NumPy**: On macOS Intel with TF 2.12, pin `numpy<2` in the env YAML (already done). If you see `_ARRAY_API not found`, you’re on NumPy 2.x with TF compiled against 1.x — recreate the env.
- **TF/CPU instructions**: Logs mentioning SSE/AVX are informational. For CPU-only builds, this is expected.
- **PYTHONPATH**: If imports fail (e.g., `ModuleNotFoundError`), confirm `PYTHONPATH=src` is set in the shell where you run `pytest`.

---

## 📐 Design principles
- **No plugs, no circularity**: Interest computed on **prior** balances; financing is explicit; excess cash → ST investments; deficits → ST/LT borrow per policy; equity only from retained earnings.
- **Broadcast-safe**: All `[B,1]` previous balances are broadcast to `[B,T,1]` before arithmetic.
- **Identity guardrail**: Unit tests check `Assets == Liabilities + Equity` to within a tight tolerance.

---

## 🔧 Configuration
Use `Config` to save/load experiment configs in JSON/YAML:
```python
from balance_sheet_forecaster.config import Config
cfg = Config()
cfg.save("runs/latest/config.yaml")
# later
cfg2 = Config.load("runs/latest/config.yaml")
```

---

## 🧪 Data
- `YahooFinancialsLoader` pulls quarterly income, balance, and cash flow statements and builds model tensors. Consider using `DummyData` for unit tests and CI.

---

## 📄 License
TBD — MIT/Apache-2.0 recommended for open source.

---

## 🤝 Contributing
PRs welcome. Please keep unit tests passing and add tests for new structural changes.
