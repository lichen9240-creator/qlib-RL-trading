"""Compare the reconstructed initial OPDS policy with trained checkpoints."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import qlib
import tianshou
import torch
import yaml
from qlib.rl.contrib import train_onpolicy
from qlib.rl.contrib.backtest import backtest
from qlib.rl.contrib.naive_config_parser import get_backtest_config_fromfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = [
    ("initialized", "Training before, reconstructed seed 42", 0),
    ("validation_best", "Validation-selected checkpoint", 12),
    ("final", "Final checkpoint", 212),
    ("twap", "TWAP benchmark", None),
]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_policy_state(config):
    captured = {}

    def capture_policy(**kwargs):
        captured.update(
            (key, value.detach().cpu().clone())
            for key, value in kwargs["policy"].state_dict().items()
        )

    original = train_onpolicy.train_and_test
    train_onpolicy.train_and_test = capture_policy
    try:
        train_onpolicy.main(deepcopy(config), run_training=True, run_backtest=False)
    finally:
        train_onpolicy.train_and_test = original
    assert captured, "The official initialization did not produce model weights"
    assert all(torch.isfinite(value).all() for value in captured.values())
    return captured


def reconstruct_initial_policy(config):
    assert config["runtime"]["seed"] == 42
    # Qlib initializes its tensors on CPU before moving the policy to CUDA.
    if not torch.cuda.is_available():
        config = deepcopy(config)
        config["runtime"]["use_cuda"] = False
    first = initial_policy_state(config)
    second = initial_policy_state(config)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)
    path = ROOT / "models/000.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(first, path)
    return {
        "method": "Reconstructed with the original seed and official train_onpolicy.main constructor order",
        "seed": 42,
        "original_initial_checkpoint_saved": False,
        "training_executed_during_reconstruction": False,
        "independent_reconstructions_identical": True,
        "state_tensor_count": len(first),
        "checkpoint": "models/000.pth",
        "sha256": file_hash(path),
    }


def expected_order_index():
    orders = pd.read_pickle(ROOT / "data/orders/test_orders.pkl")
    expected = pd.MultiIndex.from_arrays(
        [
            orders.index.get_level_values("instrument"),
            pd.to_datetime(orders.index.get_level_values("date")),
        ],
        names=["instrument", "datetime"],
    ).sort_values()
    assert len(expected) == 540 and expected.is_unique
    return expected


def metric_columns(frame):
    numeric = frame.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all(), "Non-finite backtest output"
    assert frame["ffr"].between(0, 1 + 1e-8).all()
    assert frame["trade_cost"].ge(0).all()
    notional = frame["deal_amount"].abs() * frame["base_price"]
    assert notional.gt(0).all(), "Orders without a fill need separate fee handling"
    fee_bps = frame["trade_cost"] / notional * 10000
    return pd.DataFrame(
        {
            "pa_bps": frame["pa"],
            "trade_cost": frame["trade_cost"],
            "deal_amount": frame["deal_amount"],
            "base_price": frame["base_price"],
            "fee_bps": fee_bps,
            "net_pa_bps": frame["pa"] - fee_bps,
            "ffr": frame["ffr"],
        }
    )


def summarize(columns):
    notional = columns["deal_amount"].abs() * columns["base_price"]
    return {
        "orders": len(columns),
        "mean_pa_bps": float(columns["pa_bps"].mean()),
        "median_pa_bps": float(columns["pa_bps"].median()),
        "notional_weighted_pa_bps": float(np.average(columns["pa_bps"], weights=notional)),
        "mean_fee_bps": float(columns["fee_bps"].mean()),
        "mean_fee_adjusted_pa_bps": float(columns["net_pa_bps"].mean()),
        "mean_fulfillment_ratio": float(columns["ffr"].mean()),
        "minimum_fulfillment_ratio": float(columns["ffr"].min()),
        "zero_fill_orders": int(columns["ffr"].eq(0).sum()),
    }


def main():
    os.chdir(ROOT)
    RESULTS.mkdir(exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/train_opds.yml").read_text(encoding="utf-8"))
    reconstruction = reconstruct_initial_policy(config)
    status_path = ROOT / "provenance/training_status.json"
    original_path = ROOT / "provenance/original_backtest_summary.json"
    training_status = json.loads(status_path.read_text(encoding="utf-8"))
    original_summary = json.loads(original_path.read_text(encoding="utf-8"))
    assert training_status["phase"] == "complete"
    assert training_status["completed_iterations"] == 212
    validation = pd.read_csv(RESULTS / "validation_result.csv")
    selected_iteration = (int(validation["val/pa"].to_numpy().argmax()) + 1) * 4
    assert selected_iteration == 12
    assert file_hash(ROOT / "data/orders/test_orders.pkl") == original_summary["test_orders_sha256"]
    expected = expected_order_index()
    records = {}
    paired = []
    checkpoint_hashes = {}
    exchange_parameters = None
    shared_signature = None
    for name, label, iteration in MODELS:
        filename = "backtest_twap.yml" if iteration is None else "backtest_opds.yml"
        backtest_config = yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        backtest_config["concurrency"] = 1
        backtest_config["output_dir"] = f"results/{name}"
        if iteration is not None:
            weights = Path(f"models/{iteration:03d}.pth")
            checkpoint_hashes[name] = file_hash(ROOT / weights)
            backtest_config["strategies"]["1day"]["kwargs"]["policy"]["kwargs"]["weight_file"] = weights.as_posix()
        output = RESULTS / name
        output.mkdir(exist_ok=True)
        config_path = output / "config.yml"
        # Qlib's parser uses the Windows locale encoding when opening YAML.
        config_path.write_text(yaml.safe_dump(backtest_config), encoding="utf-8")
        parsed = get_backtest_config_fromfile(str(config_path))
        signature = {
            key: parsed[key]
            for key in ("order_file", "start_time", "end_time", "data_granularity", "qlib", "exchange")
        }
        if shared_signature is None:
            shared_signature = deepcopy(signature)
            exchange_parameters = deepcopy(parsed["exchange"])
        assert signature == shared_signature, "Backtests use different market or fee settings"
        train_onpolicy.seed_everything(42)
        started = perf_counter()
        backtest(parsed)
        frame = pd.read_csv(output / "backtest_result.csv", index_col=[0, 1], parse_dates=[1]).sort_index()
        assert frame.index.is_unique and frame.index.equals(expected), "Test order identity mismatch"
        columns = metric_columns(frame)
        record = summarize(columns)
        record.update(label=label, iteration=iteration, seconds=round(perf_counter() - started, 3))
        baseline = records.get("initialized", record)
        record["mean_pa_change_vs_initialized_bps"] = record["mean_pa_bps"] - baseline["mean_pa_bps"]
        record["mean_net_pa_change_vs_initialized_bps"] = (
            record["mean_fee_adjusted_pa_bps"] - baseline["mean_fee_adjusted_pa_bps"]
        )
        records[name] = record
        paired.append(columns.add_prefix(name + "_"))
        print(name, json.dumps(record), flush=True)
    combined = pd.concat(paired, axis=1)
    assert combined.index.equals(expected) and not combined.isna().any().any()
    combined.to_csv(RESULTS / "paired_orders.csv", encoding="utf-8-sig")
    pd.DataFrame.from_dict(records, orient="index").rename_axis("model").reset_index().to_csv(
        RESULTS / "comparison.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "models": records,
        "provenance": {
            "initialization": reconstruction,
            "environment": {
                "python": platform.python_version(),
                "qlib": qlib.__version__,
                "torch": torch.__version__,
                "tianshou": tianshou.__version__,
            },
            "checkpoint_sha256": checkpoint_hashes,
            "test_orders_sha256": file_hash(ROOT / "data/orders/test_orders.pkl"),
            "train_config_sha256": file_hash(ROOT / "configs/train_opds.yml"),
            "original_training_status": training_status,
            "original_time_accounting": original_summary["time_accounting"],
            "original_status_file": str(status_path.relative_to(ROOT)),
            "original_summary_file": str(original_path.relative_to(ROOT)),
            "validation_selection": "Maximum validation PA; validation every 4 rounds",
            "best_validation_iteration": selected_iteration,
            "exchange": exchange_parameters,
            "test_stocks": 9,
            "test_orders": len(expected),
            "metric_units": "PA and fee-adjusted PA are basis points; FFR is a ratio",
            "training_shutdown_warning": original_summary.get("shutdown_warning"),
        },
    }
    (RESULTS / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved results/comparison.json, comparison.csv and paired_orders.csv", flush=True)


if __name__ == "__main__":
    main()
