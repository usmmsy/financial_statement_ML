from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List

import json
import os
import argparse

# (Optional) YAML
try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


@dataclass
class Config:
    
    # Repo & bookkeeping

    seed: int = 42
    output_dir: str = "runs/latest"
    use_tensorboard: bool = False   # whether to log to TensorBoard, default False

    # Data / splits

    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    horizon_quarters: int = 8   # forecast horizon length
    t_holdout_last: int = 4 # last T steps held out for time-validation
    train_firms: Optional[List[str]] = None # if set, restrict training to these firms
                                            # if None, use first N after loader filtering
    train_firms_count: int = 3  # if train_firms is None, use this many firms for training

    # Model & optimization

    hidden: int = 64  # GRU hidden size in DriverHead
    lr: float = 1e-3  # learning rate
    steps: int = 500  # training steps
    optimizer: str = "adam"  # optimizer type: "adam", "sgd", etc.
    grad_clip_norm: Optional[float] = None  # if set, clip gradients by global norm
    weight_decay: float = 0.0  # L2 weight decay coefficient; e.g., set > 0 if AdamW

    # Loss weights / regularization

    log_every: int = 100  # log metrics every N steps
    eval_every: int = 100  # run evaluation every N steps
    save_every: int = 0  # save model checkpoint every N steps, default 0 = only at end

    # Device / perf knobs

    mixed_precision: bool = False  # whether to enable TF mixed precision training

    # Convenience
    notes: str = ""  # optional notes to save with config

    # Methods

    def ensure_valid(self):
        if not isinstance(self.tickers, list) or len(self.tickers) == 0:
            raise ValueError("Config.tickers must be a non-empty list of ticker strings.")
        if self.horizon_quarters <= 0:
            raise ValueError("Config.horizon_quarters must be positive.")
        if self.t_holdout_last < 0 or self.t_holdout_last >= self.horizon_quarters:
            raise ValueError("Config.t_holdout_last must be in [0, horizon_quarters).")
        if self.hidden <= 0:
            raise ValueError("Config.hidden must be positive.")
        if self.lr <= 0.0:
            raise ValueError("Config.lr must be positive.")
        if self.steps <= 0:
            raise ValueError("Config.steps must be positive.")
        if self.train_firms is not None:
            missing = [f for f in self.train_firms if f not in self.tickers]
            if missing:
                raise ValueError(f"Config.train_firms contains tickers not in Config.tickers: {missing}")
            
    # Serialization

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.lower().endswith(('.yaml', '.yml')):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML is not installed, save as JSON instead or install PyYAML.")
            with open(path, 'w') as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        else:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> 'Config':
        with open(path, 'r') as f:
            if path.lower().endswith(('.yaml', '.yml')):
                if not _HAS_YAML:
                    raise RuntimeError("PyYAML is not installed, load a .json file instead or install PyYAML.")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        config = Config(**data)
        config.ensure_valid()
        return config
    
    # (Optional) CLI helper

    @staticmethod
    def from_cli(argv: Optional[List[str]] = None) -> 'Config':
        parser = argparse.ArgumentParser(description="Balance Sheet Forecaster Config")
        parser.add_argument("--config", type=str, default="", help="Path to .json/.yml config file")
        parser.add_argument("--ticks", nargs="*", default=None, help="Override tickers")
        parser.add_argument("--horizon", type=int, default=None, help="Override horizon_quarters")
        parser.add_argument("--thold", type=int, default=None, help="Override t_holdout_last")
        parser.add_argument("--train_firms", nargs="*", default=None, help="Explicit training firms subset")
        parser.add_argument("--train_firms_count", type=int, default=None)
        parser.add_argument("--hidden", type=int, default=None)
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--w_acct", type=float, default=None)
        parser.add_argument("--w_smooth", type=float, default=None)
        parser.add_argument("--out_dir", type=str, default=None)
        parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
        parser.add_argument("--notes", type=str, default=None)
        # Add more overrides as needed

        args = parser.parse_args(argv)

        if args.config:
            config = Config.load(args.config)
        else:
            config = Config()

        # Apply overrides if provided
        if args.ticks is not None and len(args.ticks) > 0:
            config.tickers = list(args.ticks)
        if args.horizon is not None:
            config.horizon_quarters = int(args.horizon)
        if args.thold is not None:
            config.t_holdout_last = int(args.thold)
        if args.train_firms is not None and len(args.train_firms) > 0:
            config.train_firms = list(args.train_firms)
        if args.train_firms_count is not None:
            config.train_firms_count = int(args.train_firms_count)
        if args.hidden is not None:
            config.hidden = int(args.hidden)
        if args.lr is not None:
            config.lr = float(args.lr)
        if args.steps is not None:
            config.steps = int(args.steps)
        if args.w_acct is not None:
            config.w_acct = float(args.w_acct)
        if args.w_smooth is not None:
            config.w_smooth = float(args.w_smooth)
        if args.out_dir is not None:
            config.out_dir = str(args.out_dir)
        if args.tb:
            config.use_tensorboard = True
        if args.notes is not None:
            config.notes = args.notes

        config.ensure_valid()
        return config
