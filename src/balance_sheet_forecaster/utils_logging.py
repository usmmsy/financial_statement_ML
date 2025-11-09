from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import asdict, is_dataclass

import os
import json
import random
import time
import contextlib
from matplotlib.pyplot import step
import numpy as np
import tensorflow as tf


# Reproducibility helper

def set_all_seeds(seed: int = 42) -> None:
    """
    Set Python, Numpy, TF seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# Timing helper

@contextlib.contextmanager
def time_block(name: str):
    """
    Print elapsed wall time for a code block.
    """

    t0 = time.time()

    try:
        yield
    finally:
        di = time.time() - t0   # elapsed time
        print(f"[TIMER] {name}: {di:.3f} sec")


# JSON utils

def _to_jsonable(obj: Any) -> Any:
    """
    Convert dataclass or nested structure to JSON-serializable dict/list.
    """

    if is_dataclass(obj):
        return asdict(obj)
    # Handle numpy / TF types
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (tf.Tensor,)):
        try:
            return obj.numpy().tolist()
        except Exception:
            return str(obj)
    # plain types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # dict or list
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # fallback
    return str(obj)


# Run logger

class RunLogger:
    """
    Lightweight file logger + optional TensorBoard

    Files written to out_dir:
        1. meta.json: config + data provenance
        2. metrics.jsonl: line-delimited JSON of logged metrics per step
        3. events.*: TensorBoard logs (if enabled)
    """

    def __init__(self, out_dir: str, use_tensorboard: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.meta_path = os.path.join(out_dir, "meta.json")
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self._tb = tf.summary.create_file_writer(out_dir) if use_tensorboard else None
        
    # meta
    def write_meta(
            self,
            config_obj: Any,
            tickers_used: List[str],
            extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist run metadata at the start of training:
            1. config (dataclass or dict)
            2. tickers actually used after filtering
            3. any extras: horizon, split sizes, notes, git hash, etc.
        """

        meta = {
            "config": _to_jsonable(config_obj),
            "tickers_used": tickers_used,
        }
        if extras:
            meta.update(_to_jsonable(extras))
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    # metrics 
        
    def log_metric(self, step: int, split: str, **metrics: float) -> None:
        """
        Append a metrics record to metrics.jsonl and TensorBoard (if enabled).
        Example: logger.log_metric(step=100, split='train', loss=0.123, mae=0.045)
        """

        rec = {
            "step": int(step),
            "split": str(split),
        }
        for k, v in metrics.items():
            rec[k] = _to_jsonable(v)

        # JSONL
        with open(self.metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')

        # TensorBoard (optional)
        if self._tb:
            with self._tb.as_default():
                for k, v in metrics.items():
                    # ensure scalar for TB
                    try:
                        vv = float(v) if not isinstance(v, (list, dict)) else None
                    except Exception:
                        vv = None
                    if vv is not None:
                        tf.summary.scalar(f"{split}/{k}", vv, step=step)
            
    def log_dict(self, step: int, split: str, metrics: Dict[str, Any]) -> None:
        """Same as log_metric but takes a dict."""
        self.log_metric(step, split, **metrics)

    def flush(self) -> None:
        """Flush TensorBoard writer (if enabled)."""
        if self._tb:
            self._tb.flush()

    # convenience

    def print_log(self, step: int, split: str, **metrics: float) -> None:
        """Print a one-liner to stdout and log the same metrics."""
        pieces = [f"step={step}", f"split={split}"] + [f"{k}={metrics[k]:.6f}" for k in metrics]
        print(" | ".join(pieces))
        self.log_metric(step, split, **metrics)

