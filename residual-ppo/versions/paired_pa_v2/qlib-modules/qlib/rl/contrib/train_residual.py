# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Opt-in three-module experiment, leaving train_onpolicy and its metrics intact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML

from qlib.rl.order_execution.interpreter import FullHistoryStateInterpreter, CategoricalActionInterpreter
from qlib.rl.order_execution.network import Recurrent
from qlib.rl.order_execution.policy import PPO
from qlib.rl.order_execution.regime import BayesianRegime, causal_features
from qlib.rl.order_execution.residual_data import ParquetStore, ParquetOrderDataset, ParquetProcessedDataProvider, FrameSimulator
from qlib.rl.order_execution.residual_policy import ResidualPPO, ResidualTrainingVessel
from qlib.rl.order_execution.residual_simulator import (
    BASE_REWARDS, SHAPING_MODES, FrozenHighPolicy, ResidualExecutionSimulator, ResidualStateInterpreter,
    ResidualActionInterpreter, PotentialShapingReward, ResidualSimulatorFactory,
)
from qlib.rl.order_execution.reward import PPOReward
from qlib.rl.trainer import Checkpoint, TrainingVessel
from qlib.rl.trainer.callbacks import Callback, MetricsWriter
from qlib.rl.utils.log import CsvWriter, LogWriter
from qlib.rl.utils.parallel_runtime import ParallelExecutionTrainer


RUNTIME_DEFAULTS = dict(parallel_mode="subproc", worker_threads=1, worker_timeout=120,
                        file_cache_size=256, day_cache_size=4096, regime_cache_size=4096,
                        high_batch_size=8, high_batch_wait_ms=0.5, gru_sequence=True, profile_execution=False)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def load_config(path, overrides=()):
    yaml = YAML(typ="safe")
    with open(path) as stream:
        config = yaml.load(stream)
    for key, value in RUNTIME_DEFAULTS.items():
        config["runtime"].setdefault(key, value)
    for override in overrides:
        key, value = override.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node and key not in ("base_reward", "shaping.mode"):
            raise ValueError(f"Unknown hyperparameter: {key}")
        node[parts[-1]] = yaml.load(value)
    for module in ("regime", "residual", "shaping"):
        if type(config[module]["enabled"]) is not bool:
            raise ValueError(f"{module}.enabled must be a YAML boolean")
    if config.get("base_reward", "ppo_discrete") not in BASE_REWARDS:
        raise ValueError(f"base_reward must be one of {BASE_REWARDS}")
    shaping = config["shaping"]
    if shaping.get("mode", "schedule") not in SHAPING_MODES:
        raise ValueError(f"shaping.mode must be one of {SHAPING_MODES}")
    if shaping.get("mode") == "inventory_mtm":
        if config.get("base_reward") != "arrival_bps" or not 0 <= shaping["coefficient"] <= 1:
            raise ValueError("Inventory shaping requires arrival_bps and a weight between 0 and 1")
    if shaping.get("mode") == "relative_inventory_mtm":
        if config.get("base_reward") != "window_pa" or not 0 <= shaping["coefficient"] <= 1:
            raise ValueError("Relative inventory shaping requires window_pa and a weight between 0 and 1")
    if config["runtime"]["concurrency"] < 1:
        raise ValueError("concurrency must be positive")
    if config["runtime"]["parallel_mode"] not in ("dummy", "subproc"):
        raise ValueError("runtime.parallel_mode must be dummy or subproc")
    for key in ("gru_sequence", "profile_execution"):
        if type(config["runtime"][key]) is not bool:
            raise ValueError(f"runtime.{key} must be a YAML boolean")
    if type(config["runtime"]["high_batch_size"]) is not int or config["runtime"]["high_batch_size"] < 1:
        raise ValueError("runtime.high_batch_size must be a positive integer")
    wait_ms = config["runtime"]["high_batch_wait_ms"]
    if type(wait_ms) not in (int, float) or not math.isfinite(wait_ms) or wait_ms < 0:
        raise ValueError("runtime.high_batch_wait_ms must be finite and nonnegative")
    for key in ("worker_threads", "worker_timeout"):
        if config["runtime"][key] < 1:
            raise ValueError(f"runtime.{key} must be positive")
    for key in ("file_cache_size", "day_cache_size", "regime_cache_size"):
        if type(config["runtime"][key]) is not int or config["runtime"][key] < 0:
            raise ValueError(f"runtime.{key} must be a nonnegative integer")
    if config["ppo"]["discount_factor"] != 1:
        raise ValueError("gamma=1 is required for the undiscounted order objective")
    if Path(config["regime"]["model_path"]).suffix != ".npz":
        raise ValueError("regime.model_path must end in .npz")
    for key in ("iterations", "episodes_per_collect", "buffer_size", "batch_size", "repeat", "validate_every"):
        if config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    for key in ("train_limit", "valid_limit", "test_limit"):
        if config["training"][key] is not None and config["training"][key] < 1:
            raise ValueError(f"training.{key} must be null or positive")
    return config


def dataset(config, store, split, limit=None):
    return ParquetOrderDataset(Path(config["data"]["orders"]) / split, store,
                               config["data"]["start_index"], config["data"]["end_index"], limit)


def market_store(config):
    options = {key: config["runtime"].get(key, RUNTIME_DEFAULTS[key])
               for key in ("file_cache_size", "day_cache_size", "regime_cache_size")}
    return ParquetStore(config["data"]["parquet_root"], **options)


def fit_regime(config):
    spec = config["regime"]
    target = Path(spec["model_path"])
    if target.exists():
        raise FileExistsError(f"Choose a new model_path rather than overwriting frozen artifact: {target}")
    store = market_store(config)
    rows = dataset(config, store, "train").rows[["instrument", "date"]].drop_duplicates()
    rows["date"] = pd.to_datetime(rows.date)
    if rows.empty or rows.date.max() > pd.Timestamp(config["data"]["train_end"]):
        raise ValueError("Training orders exceed the configured training cutoff")
    if spec["max_sequences"] is not None and len(rows) > spec["max_sequences"]:
        rows = rows.sample(spec["max_sequences"], random_state=config["runtime"]["seed"])
    sequences = [causal_features(store.day("backtest", row.instrument, row.date)) for row in rows.itertuples()]
    model = BayesianRegime(n_states=spec["n_states"], transition_prior=spec["transition_prior"],
                           sticky_prior=spec["sticky_prior"], variance_floor=spec["variance_floor"],
                           seed=config["runtime"]["seed"])
    model.fit(sequences, spec["max_iter"], spec["tol"], metadata={
        "train_start": str(rows.date.min()), "train_end": str(rows.date.max()),
        "training_keys_sha256": hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest(),
        "parquet_root": str(store.root),
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    print(json.dumps(dict(artifact=str(target), negative_elbo=model.losses, **model.metadata), indent=2))


def build(config):
    seed = config["runtime"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["runtime"]["torch_threads"])
    torch.backends.cudnn.deterministic = True
    device = config["runtime"]["device"]
    store = market_store(config)
    model = BayesianRegime.load(config["regime"]["model_path"])
    if model.n_states != config["regime"]["n_states"]:
        raise ValueError("Regime artifact state count differs from configuration")
    if pd.Timestamp(model.metadata["train_end"]) > pd.Timestamp(config["data"]["train_end"]):
        raise ValueError("Regime artifact was fitted beyond the training cutoff")
    with open(config["high_level"]["config"]) as stream:
        high_config = YAML(typ="safe").load(stream)
    if high_config["simulator"]["vol_limit"] is not None:
        raise ValueError("This residual experiment supports the existing uncapped Simple baseline only")
    if high_config["policy"]["class"] != "PPO" or high_config["network"]["class"] != "Recurrent":
        raise ValueError("Expected the original Recurrent PPO baseline")
    if high_config["reward"]["class"] != "PPOReward":
        raise ValueError("Expected the unchanged PPOReward baseline")
    if high_config["state_interpreter"]["class"] != "FullHistoryStateInterpreter" or high_config["action_interpreter"]["class"] != "CategoricalActionInterpreter":
        raise ValueError("Expected the original full-history state and categorical high-level actions")
    state_kwargs = copy.deepcopy(high_config["state_interpreter"]["kwargs"])
    provider_kwargs = state_kwargs["processed_data_provider"]["kwargs"]
    state_kwargs["processed_data_provider"] = ParquetProcessedDataProvider(
        store, provider_kwargs["feature_columns_today"], provider_kwargs["feature_columns_yesterday"])
    high_state = FullHistoryStateInterpreter(**state_kwargs)
    high_action = CategoricalActionInterpreter(**high_config["action_interpreter"]["kwargs"])
    if state_kwargs["max_step"] != high_config["reward"]["kwargs"]["max_step"] or state_kwargs["max_step"] != high_config["action_interpreter"]["kwargs"]["max_step"]:
        raise ValueError("High-level state, reward and action horizons must match")
    high_network = Recurrent(high_state.observation_space, **high_config["network"].get("kwargs", {}))
    high_policy = PPO(high_network, high_state.observation_space, high_action.action_space,
                      **high_config["policy"]["kwargs"], weight_file=Path(config["high_level"]["checkpoint"])).to(device)
    high = FrozenHighPolicy(high_policy, high_state, high_action)
    residual = config["residual"]
    state = ResidualStateInterpreter(residual["window"], model.n_states)
    action = ResidualActionInterpreter(len(residual["values"]))
    policy = ResidualPPO(state.observation_space, action.action_space, model.n_states,
                         zero_action=residual["values"].index(0), hidden_dim=residual["hidden_dim"],
                         private_dim=residual["private_dim"],
                         gru_sequence=config["runtime"].get("gru_sequence", True), **config["ppo"]).to(device)
    reward = PotentialShapingReward(config["shaping"]["enabled"], config["ppo"]["discount_factor"])
    simulator_kwargs = dict(high_policy=high, regime=model, regime_enabled=config["regime"]["enabled"],
        residual_enabled=residual["enabled"], residual_scale=residual["scale"], residual_values=residual["values"],
        base_reward=config.get("base_reward", "ppo_discrete"),
        shaping_mode=config["shaping"].get("mode", "schedule"),
        shaping_coefficient=config["shaping"]["coefficient"], window=residual["window"],
        macro_minutes=high_config["simulator"]["time_per_step"],
        data_granularity=high_config["simulator"]["data_granularity"], max_step=state_kwargs["max_step"])

    factory = ResidualSimulatorFactory(store, simulator_kwargs)
    return store, high, model, state, action, policy, reward, factory, simulator_kwargs


class ExecutionProgress(LogWriter):
    """Rate-limited console heartbeat, separate from unchanged result metrics."""
    def __init__(self):
        super().__init__()
        self.phase = "Execution"
        self.target = 0
        self.started = self.last_print = time.monotonic()
        self.first_episode = self.first_step = 0
        self.residual_decisions = self.nonzero_actions = self.changed_amounts = 0

    def begin(self, phase, target):
        self.phase, self.target = phase, target
        self.started = self.last_print = time.monotonic()
        self.first_episode, self.first_step = self.global_episode, self.global_step
        self.residual_decisions = self.nonzero_actions = self.changed_amounts = 0
        print(f"[{phase}] Starting {target} orders", flush=True)

    def report(self):
        now = time.monotonic()
        print(f"[{self.phase}] {self.global_episode - self.first_episode}/{self.target} orders, "
              f"{self.global_step - self.first_step} steps, {now - self.started:.1f}s", flush=True)
        self.last_print = now

    def log_step(self, reward, contents):
        if time.monotonic() - self.last_print >= 10:
            self.report()

    def log_episode(self, length, rewards, contents):
        terminal = contents[-1]
        self.residual_decisions += int(terminal["residual_decisions"])
        self.nonzero_actions += int(terminal["nonzero_residual_actions"])
        self.changed_amounts += int(terminal["changed_residual_amounts"])
        if self.global_episode - self.first_episode == self.target:
            self.report()
            print(f"[Residual actions] {self.phase}: eligible={self.residual_decisions}, "
                  f"nonzero={self.nonzero_actions}, changed_amount={self.changed_amounts}", flush=True)


class StageProgress(Callback):
    def __init__(self, writer):
        self.writer = writer

    def on_train_start(self, trainer, vessel):
        self.writer.begin(f"Iter {trainer.current_iter + 1}/{trainer.max_iters}: collect", vessel.episode_per_iter)

    def on_train_end(self, trainer, vessel):
        print(f"[Train complete] PA={trainer.metrics.get('pa', float('nan')):.6f}, "
              f"FFR={trainer.metrics.get('ffr', float('nan')):.6f}", flush=True)

    def on_validate_start(self, trainer, vessel):
        self.writer.begin(f"Iter {trainer.current_iter + 1}/{trainer.max_iters}: validation", len(vessel.val_initial_states))

    def on_validate_end(self, trainer, vessel):
        print(f"[Validation complete] PA={trainer.metrics.get('val/pa', float('nan')):.6f}, "
              f"FFR={trainer.metrics.get('val/ffr', float('nan')):.6f}", flush=True)


class BestValidation(Callback):
    def __init__(self, output):
        self.output, self.best = Path(output), -np.inf

    def on_validate_end(self, trainer, vessel):
        pa = trainer.metrics["val/pa"]
        if pa > self.best:
            self.best = pa
            torch.save(trainer.state_dict(), self.output / "best_validation.pth")

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        self.best = state_dict["best"]


def manifest(config):
    metadata = {
        "regime_sha256": sha256(config["regime"]["model_path"]),
        "high_sha256": sha256(config["high_level"]["checkpoint"]),
        "high_config_sha256": sha256(config["high_level"]["config"]),
        "architecture": {k: config["residual"][k] for k in ("window", "hidden_dim", "private_dim", "values")},
        "n_states": config["regime"]["n_states"],
    }
    # A missing reward in legacy checkpoints denotes the discrete PPO objective.
    if config.get("base_reward", "ppo_discrete") != "ppo_discrete":
        metadata["base_reward"] = config["base_reward"]
    if "mode" in config["shaping"]:
        metadata["shaping"] = dict(config["shaping"])
    return metadata


def check_parity(config, limit):
    if limit < 1:
        raise ValueError("Parity limit must be positive")
    store, high, _, _, _, _, reward, _, kwargs = build(config)
    kwargs.update(residual_enabled=False)
    results = []
    for i, order in enumerate(dataset(config, store, "valid", limit)):
        frame = store.day("backtest", order.stock_id, order.start_time)
        original = FrameSimulator(order, frame, kwargs["macro_minutes"], kwargs["data_granularity"])
        raw_reward = PPOReward(kwargs["max_step"]) if kwargs["base_reward"] == "ppo_discrete" else None
        original_return = 0.0
        while not original.done():
            original.step(high(original.get_state()))
            if raw_reward is not None:
                original_return += raw_reward.reward(original.get_state())
        changed = ResidualExecutionSimulator(order, frame, **kwargs)
        shaped_return = 0.0
        while not changed.done():
            changed.step(0)  # deliberately nonzero residual index; disabled switch must override it
            shaped_return += reward.reward(changed.get_state())
        np.testing.assert_allclose(original.history_exec.deal_amount, changed.inner.history_exec.deal_amount, rtol=1e-10, atol=1e-7)
        np.testing.assert_allclose(original.history_steps.position, changed.macro_history.position, rtol=1e-10, atol=1e-7)
        if kwargs["base_reward"] == "arrival_bps":
            arrival = float(original.backtest_data.get_deal_price().at[original.ticks_for_order[0]])
            advantage = -order.sign * 10_000 * (original.metrics["trade_price"] / arrival - 1)
            expected_return = original.metrics["ffr"] * advantage
        else:
            expected_return = original_return
        np.testing.assert_allclose(expected_return, shaped_return, atol=1e-9)
        for key in ("pa", "ffr", "trade_price", "trade_value"):
            np.testing.assert_allclose(original.metrics[key], changed.inner.metrics[key], rtol=1e-10, atol=1e-7)
        results.append(dict(order=i, pa=original.metrics["pa"], ffr=original.metrics["ffr"]))
    print(json.dumps(dict(parity_orders=len(results), passed=True, metrics=pd.DataFrame(results).mean().to_dict())))


def run(config, command, checkpoint=None, split="valid"):
    print("[Setup] Loading market data, frozen high-level PPO and regime artifact...", flush=True)
    store, _, _, state, action, policy, reward, factory, _ = build(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.yml"
    yaml = YAML(typ="safe")
    if resolved.exists():
        with resolved.open() as stream:
            if yaml.load(stream) != config:
                raise ValueError("Output directory belongs to another configuration; choose a new output")
    with resolved.open("w") as stream:
        yaml.dump(config, stream)
    metadata = manifest(config)
    (output / "artifacts.json").write_text(json.dumps(metadata, indent=2))
    if Path(config["regime"]["model_path"]).resolve() != (output / "regime.npz").resolve():
        shutil.copyfile(config["regime"]["model_path"], output / "regime.npz")
    if checkpoint:
        saved = torch.load(checkpoint, map_location=config["runtime"]["device"], weights_only=False)
        if saved["vessel"]["artifacts"] != metadata:
            raise ValueError("Checkpoint reward, architecture or frozen artifacts do not match")
        policy.load_state_dict(saved["vessel"]["policy"])
    train_spec = config["training"]
    runtime = config["runtime"]
    execution_kwargs = dict(execution_mode=runtime["parallel_mode"], worker_threads=runtime["worker_threads"],
                            worker_timeout=runtime["worker_timeout"], concurrency=runtime["concurrency"],
                            high_batch_size=runtime["high_batch_size"], high_batch_wait_ms=runtime["high_batch_wait_ms"],
                            profile_execution=runtime["profile_execution"])
    print(f"[Runtime] {runtime['parallel_mode']}, {runtime['concurrency']} environments; "
          f"file/day/regime cache={runtime['file_cache_size']}/{runtime['day_cache_size']}/{runtime['regime_cache_size']}",
          flush=True)
    print(f"[Inference] high_batch_size={runtime['high_batch_size']}, "
          f"high_batch_wait_ms={runtime['high_batch_wait_ms']}, gru_sequence={runtime['gru_sequence']}", flush=True)
    enabled = config["residual"]["enabled"] and config["residual"]["scale"] > 0
    print(f"[Low-level reward] base={config.get('base_reward', 'ppo_discrete')}, "
          f"shaping={config['shaping']['enabled']}, mode={config['shaping'].get('mode', 'schedule')}, "
          f"weight={config['shaping']['coefficient']}", flush=True)
    if command == "train" and enabled:
        if checkpoint:
            raise ValueError("--checkpoint is evaluation-only; start each ablation from the same initialization")
        if (output / "checkpoints").exists():
            raise FileExistsError("Training output already exists; choose a new output directory")
        episodes, concurrency = train_spec["episodes_per_collect"], config["runtime"]["concurrency"]
        # Episodes can have very different lengths. A busy worker must not evict
        # its earlier transitions before the one on-policy update consumes them.
        if train_spec["buffer_size"] // concurrency < (episodes + concurrency) * 48:
            raise ValueError("Replay buffer too small to retain all micro transitions in a collect")
        train_data = dataset(config, store, "train", train_spec["train_limit"])
        if pd.to_datetime(train_data.rows.date).max() > pd.Timestamp(config["data"]["train_end"]):
            raise ValueError("Low-level PPO training orders exceed the training cutoff")
        vessel = ResidualTrainingVessel(simulator_fn=factory, state_interpreter=state, action_interpreter=action,
            policy=policy, reward=reward, artifact_metadata=metadata,
            train_initial_states=train_data,
            val_initial_states=dataset(config, store, "valid", train_spec["valid_limit"]),
            buffer_size=train_spec["buffer_size"], episode_per_iter=episodes,
            update_kwargs=dict(batch_size=train_spec["batch_size"], repeat=train_spec["repeat"]))
        progress = ExecutionProgress()
        trainer = ParallelExecutionTrainer(max_iters=train_spec["iterations"], val_every_n_iters=train_spec["validate_every"],
            **execution_kwargs, loggers=progress,
            callbacks=[StageProgress(progress), MetricsWriter(output), BestValidation(output),
                       Checkpoint(output / "checkpoints", every_n_iters=1, save_latest="copy")])
        trainer.fit(vessel)
    else:
        if command == "train":
            print("Residual disabled (or scale=0): skip PPO updates and evaluate frozen baseline.")
        if command == "evaluate" and enabled and not checkpoint:
            raise ValueError("Evaluation of a learned residual policy requires --checkpoint")
        policy.eval()
        eval_data = dataset(config, store, split, train_spec[f"{split}_limit"])
        progress = ExecutionProgress()
        progress.begin(f"Evaluate {split}", len(eval_data))
        vessel = TrainingVessel(simulator_fn=factory, state_interpreter=state, action_interpreter=action,
                                policy=policy, reward=reward, test_initial_states=eval_data)
        ParallelExecutionTrainer(**execution_kwargs, loggers=[CsvWriter(output), progress]).test(vessel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit-regime", "train", "evaluate", "check-parity"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--limit", type=int, default=32, help="Number of validation orders in parity check")
    parser.add_argument("--show-pandas-deprecation", action="store_true",
                        help="Show the known pandas 1.5 / NumPy 1.25+ find_common_type deprecation")
    args = parser.parse_args()
    # Unlike the original entry point's blanket suppression, keep other
    # deprecations and all RuntimeWarnings visible (including NaN/overflow).
    warnings.filterwarnings(
        "default" if args.show_pandas_deprecation else "ignore",
        message=r"np\.find_common_type is deprecated\.",
        category=DeprecationWarning,
        module=r"pandas\.core\.dtypes\.cast",
    )
    config = load_config(args.config, args.set)
    if args.command == "fit-regime":
        fit_regime(config)
    elif args.command == "check-parity":
        check_parity(config, args.limit)
    else:
        run(config, args.command, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
