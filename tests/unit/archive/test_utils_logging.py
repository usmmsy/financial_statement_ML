# tests/unit/test_utils_logging.py
import json
# import os
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import pytest

from balance_sheet_forecaster.utils_logging import (
    set_all_seeds,
    time_block,
    RunLogger,
    _to_jsonable,  # optional: useful for a couple of direct spot checks
)

# ---------- helpers ----------

@dataclass
class DemoCfg:
    a: int
    b: float
    c: str

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# ---------- tests ----------

def test_set_all_seeds_is_deterministic():
    def sample():
        set_all_seeds(123)
        py = random.random()
        npy = np.random.rand()
        tfy = tf.random.uniform([1], 0, 1, dtype=tf.float32).numpy()[0]
        return py, npy, float(tfy)

    a = sample()
    b = sample()
    assert a == b  # exact match across all three rngs

def test_time_block_prints_timer(capsys):
    with time_block("my-block"):
        sum(range(10_000))  # do a tiny bit of work
    out = capsys.readouterr().out
    assert "[TIMER] my-block:" in out
    assert out.strip().endswith("sec")

def test_to_jsonable_direct_spot_checks():
    # numpy scalar -> python scalar
    assert isinstance(_to_jsonable(np.float32(1.25)), float)
    # numpy array -> list
    assert _to_jsonable(np.array([1, 2], dtype=np.int32)) == [1, 2]
    # tensor -> list
    assert _to_jsonable(tf.constant([[3.0]], tf.float32)) == [[3.0]]
    # dataclass -> dict
    dj = _to_jsonable(DemoCfg(1, 2.0, "x"))
    assert dj == {"a": 1, "b": 2.0, "c": "x"}
    # tuple -> list
    assert _to_jsonable((1, 2, 3)) == [1, 2, 3]

def test_runlogger_writes_meta_and_metrics(tmp_path, capsys):
    out_dir = tmp_path / "run"
    logger = RunLogger(str(out_dir), use_tensorboard=False)

    # 1) meta.json includes config, tickers, and extras merged
    cfg = DemoCfg(a=7, b=3.5, c="note")
    extras = {"horizon": 8, "split": {"train": 3, "val": 1}}
    logger.write_meta(cfg, tickers_used=["AAPL", "MSFT"], extras=extras)

    meta_path = out_dir / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["config"] == {"a": 7, "b": 3.5, "c": "note"}
    assert meta["tickers_used"] == ["AAPL", "MSFT"]
    assert meta["horizon"] == 8
    assert meta["split"] == {"train": 3, "val": 1}

    # 2) log_metric appends JSONL with scalarized values
    logger.log_metric(step=10, split="train", loss=np.float32(0.123), mae=0.045)
    logger.log_dict(step=11, split="val", metrics={"loss": 0.5, "mae": np.array(0.2)})

    metrics_path = out_dir / "metrics.jsonl"
    lines = _read_jsonl(metrics_path)
    assert len(lines) == 2
    assert lines[0]["step"] == 10 and lines[0]["split"] == "train"
    assert pytest.approx(lines[0]["loss"], rel=1e-6, abs=1e-8) == 0.123
    assert pytest.approx(lines[0]["mae"], rel=1e-6, abs=1e-8) == 0.045
    assert lines[1]["step"] == 11 and lines[1]["split"] == "val"
    assert pytest.approx(lines[1]["loss"], rel=1e-6, abs=1e-8) == 0.5
    assert pytest.approx(lines[1]["mae"], rel=1e-6, abs=1e-8) == 0.2

    # 3) print_log prints and also appends JSONL
    logger.print_log(step=12, split="train", loss=1.0, mae=2.0)
    out = capsys.readouterr().out
    assert "step=12 | split=train | loss=1.000000 | mae=2.000000" in out

    lines = _read_jsonl(metrics_path)
    assert len(lines) == 3
    assert lines[-1]["step"] == 12 and lines[-1]["split"] == "train"

    # 4) flush is a no-op if TB disabled (should not raise)
    logger.flush()
