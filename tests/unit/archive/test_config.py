import json
import sys
import pathlib
import pytest

from balance_sheet_forecaster.config import Config

# (Optional) YAML
try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ---------- ensure_valid ----------
def test_validation_empty_tickers():
    with pytest.raises(ValueError):
        Config(tickers=[]).ensure_valid()

@pytest.mark.parametrize("hq", [0, -1])
def test_validation_bad_horizon(hq):
    with pytest.raises(ValueError):
        Config(horizon_quarters=hq).ensure_valid()

@pytest.mark.parametrize("th", [-1, 8])
def test_validation_bad_holdout(th):
    with pytest.raises(ValueError):
        Config(horizon_quarters=8, t_holdout_last=th).ensure_valid()

def test_validation_train_firms_subset():
    with pytest.raises(ValueError):
        Config(tickers=["A","B"], train_firms=["A","C"]).ensure_valid()

def test_ensure_valid_happy_path():
    cfg = Config()
    # Should not raise
    cfg.ensure_valid()

def test_ensure_valid_bad_horizon():
    cfg = Config(horizon_quarters=0)
    with pytest.raises(ValueError):
        cfg.ensure_valid()

def test_ensure_valid_bad_holdout_range():
    cfg = Config(horizon_quarters=8, t_holdout_last=8)  # must be < horizon
    with pytest.raises(ValueError):
        cfg.ensure_valid()

def test_ensure_valid_train_firms_not_in_tickers():
    cfg = Config(tickers=["AAPL", "MSFT"], train_firms=["AAPL", "GOOG"])
    with pytest.raises(ValueError):
        cfg.ensure_valid()

def test_ensure_valid_bad_hidden_lr_steps():
    with pytest.raises(ValueError):
        Config(hidden=0).ensure_valid()
    with pytest.raises(ValueError):
        Config(lr=0.0).ensure_valid()
    with pytest.raises(ValueError):
        Config(steps=0).ensure_valid()

def test_round_trip_types(tmp_path: pathlib.Path):
    path = tmp_path / "c.json"
    cfg = Config(tickers=["X"], horizon_quarters=6, lr=0.001, notes="hi")
    cfg.save(str(path))
    back = Config.load(str(path))
    assert back.tickers == ["X"]
    assert isinstance(back.lr, float)
    assert back.notes == "hi"

def test_defaults_survive_round_trip(tmp_path):
    p = tmp_path / "d.json"
    c = Config()
    c.save(str(p))
    back = Config.load(str(p))
    assert back.train_firms is None
    assert back.train_firms_count == c.train_firms_count
    assert back.output_dir == c.output_dir

def test_unknown_key_rejected(tmp_path: pathlib.Path):
    path = tmp_path / "weird.json"
    with open(path, "w") as f:
        json.dump({"tickers": ["X"], "horizon_quarters": 6, "unknown": 1}, f)
    with pytest.raises(TypeError):
        Config.load(str(path))

@pytest.mark.skipif(not _HAS_YAML, reason="PyYAML not installed")
def test_yaml_case_insensitive_suffix(tmp_path: pathlib.Path):
    p = tmp_path / "ConfIG.YML"
    Config().save(str(p))
    loaded = Config.load(str(p))
    assert isinstance(loaded, Config)

def test_save_creates_parent_dirs(tmp_path):
    deep = tmp_path / "nested" / "more" / "cfg.json"
    Config().save(str(deep))
    assert deep.exists()

# ---------- to_dict ----------

def test_to_dict_includes_core_fields():
    cfg = Config()
    d = cfg.to_dict()
    # Sanity: spot check a few fields
    assert d["seed"] == 42
    assert isinstance(d["tickers"], list)
    assert "horizon_quarters" in d
    assert "use_tensorboard" in d

# ---------- save/load (JSON) ----------

def test_json_round_trip(tmp_path: pathlib.Path):
    cfg = Config(tickers=["AAA", "BBB"], horizon_quarters=6, t_holdout_last=2)
    path = tmp_path / "config.json"
    cfg.save(str(path))

    assert path.exists()
    # Load back and compare a few fields
    loaded = Config.load(str(path))
    assert loaded.tickers == ["AAA", "BBB"]
    assert loaded.horizon_quarters == 6
    assert loaded.t_holdout_last == 2
    # ensure_valid was invoked in load()
    loaded.ensure_valid()  # should not raise

# ---------- save/load (YAML, optional) ----------

@pytest.mark.skipif("yaml" not in sys.modules and __import__("importlib").util.find_spec("yaml") is None,
                    reason="PyYAML not installed")
def test_yaml_round_trip(tmp_path: pathlib.Path):
    cfg = Config(tickers=["X", "Y", "Z"], hidden=128, notes="hello")
    path = tmp_path / "config.yml"
    cfg.save(str(path))

    loaded = Config.load(str(path))
    assert loaded.tickers == ["X", "Y", "Z"]
    assert loaded.hidden == 128
    assert loaded.notes == "hello"

# ---------- CLI parsing ----------

def test_from_cli_with_config_path_json(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    # Prepare a base JSON config
    base = Config(tickers=["T1", "T2"], horizon_quarters=8, t_holdout_last=3, notes="base")
    path = tmp_path / "base.json"
    base.save(str(path))

    # Override a few flags from CLI
    argv = ["--config", str(path), "--ticks", "AAPL", "MSFT", "--horizon", "10", "--thold", "4",
            "--hidden", "256", "--lr", "0.0005", "--steps", "123", "--tb", "--notes", "cli"]
    monkeypatch.setattr("sys.argv", ["prog"] + argv)

    cfg = Config.from_cli()
    assert cfg.tickers == ["AAPL", "MSFT"]
    assert cfg.horizon_quarters == 10
    assert cfg.t_holdout_last == 4
    assert cfg.hidden == 256
    assert abs(cfg.lr - 0.0005) < 1e-12
    assert cfg.steps == 123
    assert cfg.use_tensorboard is True
    assert cfg.notes == "cli"

def test_from_cli_without_base_uses_defaults(monkeypatch: pytest.MonkeyPatch):
    argv = ["--ticks", "A", "B", "C", "--horizon", "6", "--thold", "2"]
    monkeypatch.setattr("sys.argv", ["prog"] + argv)
    cfg = Config.from_cli()
    assert cfg.tickers == ["A", "B", "C"]
    assert cfg.horizon_quarters == 6
    assert cfg.t_holdout_last == 2

def test_from_cli_invalid_after_override_raises(monkeypatch: pytest.MonkeyPatch):
    # Horizon overridden to 4 but holdout to 5 (invalid)
    argv = ["--horizon", "4", "--thold", "5"]
    monkeypatch.setattr("sys.argv", ["prog"] + argv)
    with pytest.raises(ValueError):
        Config.from_cli()

# ---------- load invalid file ----------

def test_load_invalid_json(tmp_path: pathlib.Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not: valid json}")
    with pytest.raises(Exception):
        Config.load(str(bad))
