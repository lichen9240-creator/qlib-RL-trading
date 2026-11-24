"""Baseline-first Qlib experiment suite for the development and full datasets."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
YAML_IO = YAML(typ="safe")
EXECUTION_DEFAULTS = dict(parallel_mode="subproc", worker_threads=1, worker_timeout=120,
                          file_cache_size=256, day_cache_size=4096, regime_cache_size=4096)
RESIDUAL_EXECUTION_DEFAULTS = dict(high_batch_size=8, high_batch_wait_ms=0.5,
                                   gru_sequence=True, profile_execution=False)
PROGRESSIVE_VARIANTS = ("residual_only", "residual_regime", "full")
LEAVE_ONE_OUT_VARIANTS = ("no_regime", "no_residual", "no_shaping")
VARIANT_MODULES = {
    "residual_only": ("residual",),
    "residual_regime": ("residual", "regime"),
    "full": ("residual", "regime", "shaping"),
    "no_regime": ("residual", "shaping"),
    "no_residual": ("regime", "shaping"),
    "no_shaping": ("residual", "regime"),
}


def read_yaml(path):
    with Path(path).open() as stream:
        return YAML_IO.load(stream)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def write_config(path, value):
    path = Path(path)
    if path.exists():
        if read_yaml(path) != value:
            raise RuntimeError(f"Configuration changed; refusing to overwrite {path}")
        return
    with path.open("w") as stream:
        YAML_IO.dump(value, stream)


def validate_settings(s):
    for key, value in {**EXECUTION_DEFAULTS, **RESIDUAL_EXECUTION_DEFAULTS}.items():
        s.setdefault(key, value)
    s.setdefault("baseline_concurrency", 16)
    s.setdefault("residual_concurrency", 16)
    variants = s["variants"]
    if not isinstance(variants, list):
        raise ValueError("variants must be a list; use [] for baseline only")
    if variants:
        if "full" not in variants or len(set(variants)) != len(variants) or not set(variants) <= VARIANT_MODULES.keys():
            raise ValueError(f"variants must include full and use each model at most once: {list(VARIANT_MODULES)}")
        if s.get("residual_base_reward", "ppo_discrete") not in ("ppo_discrete", "arrival_bps"):
            raise ValueError("residual_base_reward must be ppo_discrete or arrival_bps")
        if "initial_zero_bias" in s:
            bias = s["initial_zero_bias"]
            if type(bias) not in (int, float) or not math.isfinite(bias):
                raise ValueError("initial_zero_bias must be a finite logit offset")
        if type(s["high_batch_size"]) is not int or s["high_batch_size"] < 1:
            raise ValueError("high_batch_size must be a positive integer")
        if (type(s["high_batch_wait_ms"]) not in (int, float) or not math.isfinite(s["high_batch_wait_ms"])
                or s["high_batch_wait_ms"] < 0):
            raise ValueError("high_batch_wait_ms must be finite and nonnegative")
        for key in ("gru_sequence", "profile_execution"):
            if type(s[key]) is not bool:
                raise ValueError(f"{key} must be a YAML boolean")
        if s["regime_max_sequences"] is not None and (type(s["regime_max_sequences"]) is not int or s["regime_max_sequences"] < 1):
            raise ValueError("regime_max_sequences must be null or a positive integer")
    if s["parallel_mode"] not in ("dummy", "subproc"):
        raise ValueError("parallel_mode must be dummy or subproc")
    for key in ("worker_threads", "worker_timeout"):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("file_cache_size", "day_cache_size", "regime_cache_size"):
        if type(s[key]) is not int or s[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    iteration_keys = ["baseline_iterations"]
    counts = ["episodes_per_collect", "validate_every", "baseline_concurrency", "baseline_batch_size", "baseline_repeat"]
    if "baseline_buffer_size" in s:
        counts.append("baseline_buffer_size")
    if variants:
        iteration_keys.append("residual_iterations")
        counts.extend(["residual_concurrency", "residual_batch_size", "residual_repeat", "residual_buffer_size", "regime_max_iter"])
    for key in iteration_keys + counts:
        if type(s[key]) is not int or s[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in iteration_keys:
        if s[key] % s["validate_every"]:
            raise ValueError(f"{key} must be a multiple of validate_every")
    if s["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if "iteration_stocks" in s:
        if type(s["iteration_stocks"]) is not int or s["iteration_stocks"] < 1 or not s.get("orders_dir"):
            raise ValueError("iteration_stocks needs a positive count and a separate orders_dir")
    episodes = s["episodes_per_collect"]
    # The conservative bound allows uneven episode allocation between workers.
    n = s["baseline_concurrency"]
    if s.get("baseline_buffer_size", 20000) // n < (episodes + n) * 8:
        raise ValueError("Increase baseline_buffer_size for the requested collection size")
    if variants:
        n = s["residual_concurrency"]
        if s["residual_buffer_size"] // n < (episodes + n) * 48:
            raise ValueError("Increase residual_buffer_size for the requested collection size")


def read_order_split(directory):
    import pandas as pd
    from qlib.utils.pickle_utils import restricted_pickle_load

    frames, digests = [], {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                frames.append(restricted_pickle_load(stream).reset_index())
            digests[path.name] = fingerprint(path)
    if not frames:
        raise ValueError(f"Empty order split: {directory}")
    return pd.concat(frames, ignore_index=True), digests


def inspect_orders(orders_dir):
    import pandas as pd

    result = {}
    for split in ("train", "valid", "test"):
        rows, digests = read_order_split(orders_dir / split)
        dates = pd.to_datetime(rows["date"])
        if rows.duplicated(["instrument", "date"]).any():
            raise ValueError("This suite expects one order per stock-day for paired result comparison")
        result[split] = dict(count=len(rows), stocks=int(rows.instrument.nunique()),
                             start=str(dates.min()), end=str(dates.max()), files=digests)
    if not (result["train"]["end"] < result["valid"]["start"] <= result["valid"]["end"] < result["test"]["start"]):
        raise ValueError("Train / validation / test date ranges must be disjoint and chronological")
    if result["train"]["end"] > "2021-06-30 23:59:59":
        raise ValueError("Training data exceeds the current course experiment cutoff")
    return result


class Suite:
    def __init__(self, run_dir, settings, restart_incomplete):
        self.run_dir, self.s, self.restart = run_dir, settings, restart_incomplete
        self.baseline_repo = (ROOT / settings["baseline_repo"]).resolve()
        self.residual_repo = (ROOT / settings["residual_repo"]).resolve()
        self.data_dir = (ROOT / settings["data_dir"]).resolve()
        self.orders_dir = (ROOT / settings.get("orders_dir", self.data_dir / "orders")).resolve()
        for name in ("configs", "logs", "stages", "artifacts"):
            (run_dir / name).mkdir(exist_ok=True)
        self.pipeline_log = (run_dir / "pipeline.log").open("a", buffering=1)
        self.sequence = 0

    def emit(self, message, end="\n"):
        print(message, end=end, flush=True)
        self.pipeline_log.write(message + end)

    @property
    def all_ablations(self):
        return set(LEAVE_ONE_OUT_VARIANTS) <= set(self.s["variants"])

    @property
    def progressive(self):
        return set(self.s["variants"]) == set(PROGRESSIVE_VARIANTS)

    def prepare(self):
        required = [ROOT / "scripts/parquet_adapter.py", ROOT / "scripts/original_qlib_baseline.py",
                    self.baseline_repo / "qlib/rl/contrib/train_onpolicy.py",
                    self.residual_repo / "qlib/rl/utils/parallel_runtime.py"]
        if self.s["variants"]:
            required.extend(self.residual_repo / "qlib/rl" / file for file in (
                "contrib/train_residual.py", "order_execution/residual_data.py", "order_execution/regime.py",
                "order_execution/residual_policy.py", "order_execution/residual_simulator.py"))
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        if "iteration_stocks" in self.s:
            from iteration_orders import prepare as prepare_iteration_orders

            if self.orders_dir == self.data_dir / "orders":
                raise ValueError("The iteration order directory must be separate from the full dataset")
            prepare_iteration_orders(self.data_dir / "orders", self.orders_dir, self.s["iteration_stocks"], self.s["seed"])
        orders = inspect_orders(self.orders_dir)
        manifest_path = self.run_dir / "orders_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != orders:
            raise RuntimeError("Order data changed since this experiment started")
        write_json(manifest_path, orders)
        self.test_count = orders["test"]["count"]
        self.emit("[Data] " + ", ".join(f"{name}={entry['count']}" for name, entry in orders.items()))
        self.emit(f"[Original baseline repository] {self.baseline_repo}")

        baseline = read_yaml(self.baseline_repo / "examples/rl_order_execution/exp_configs/train_ppo.yml")
        execution = {key: self.s[key] for key in EXECUTION_DEFAULTS}
        baseline["env"].update(concurrency=self.s["baseline_concurrency"],
                                **{key: value for key, value in execution.items() if key != "regime_cache_size"})
        baseline["runtime"].update(seed=self.s["seed"], use_cuda=self.s["device"] == "cuda")
        source = baseline["data"]["source"]
        source.update(order_dir=str(self.orders_dir), feature_root_dir=str(self.data_dir / "pickle"))
        baseline["state_interpreter"]["kwargs"]["processed_data_provider"]["kwargs"]["data_dir"] = str(self.data_dir / "pickle")
        baseline["policy"]["kwargs"].pop("weight_file", None)
        baseline["trainer"].update(max_epoch=self.s["baseline_iterations"],
            episode_per_collect=self.s["episodes_per_collect"], batch_size=self.s["baseline_batch_size"],
            repeat_per_collect=self.s["baseline_repeat"], val_every_n_epoch=self.s["validate_every"],
            checkpoint_path=str(self.run_dir / "baseline"), checkpoint_every_n_iters=1)
        if "baseline_buffer_size" in self.s:
            baseline["trainer"]["buffer_size"] = self.s["baseline_buffer_size"]
        baseline["trainer"].pop("earlystop_patience", None)
        # Save every iteration: upstream callback timing for every_n_iters > 1
        # is offset; exact validation-iteration weights must remain available.
        self.baseline_config = self.run_dir / "configs/baseline_train.yml"
        write_config(self.baseline_config, baseline)
        self.high_checkpoint = self.run_dir / "baseline/best_validation.pth"
        baseline_test = copy.deepcopy(baseline)
        baseline_test["policy"]["kwargs"]["weight_file"] = str(self.high_checkpoint)
        baseline_test["trainer"]["checkpoint_path"] = str(self.run_dir / "baseline_test")
        write_config(self.run_dir / "configs/baseline_test.yml", baseline_test)

        if self.s.get("baseline_source_run") and not (self.run_dir / "stages/01_reuse_baseline.json").exists():
            self.check_baseline_source(orders)
        if self.s["variants"]:
            self.prepare_residual_configs(execution)
        sources = required + [self.baseline_config, ROOT / "scripts/execution_metrics.py", Path(__file__).resolve()]
        if "iteration_stocks" in self.s:
            sources.extend([ROOT / "scripts/iteration_orders.py", self.orders_dir / "selection.json"])
        write_json(self.run_dir / "sources_latest.json", {str(p): fingerprint(p) for p in sources})
        initial_sources = self.run_dir / "sources_initial.json"
        if not initial_sources.exists():
            shutil.copyfile(self.run_dir / "sources_latest.json", initial_sources)

    def prepare_residual_configs(self, execution):
        residual = read_yaml(self.residual_repo / "examples/rl_order_execution/exp_configs/train_residual.yml")
        if "residual_base_reward" in self.s:
            residual["base_reward"] = self.s["residual_base_reward"]
        else:
            # Suites saved before continuous rewards had no key in their resolved YAML.
            residual.pop("base_reward", None)
        if "initial_zero_bias" in self.s:
            residual["ppo"]["initial_zero_bias"] = self.s["initial_zero_bias"]
        elif (self.run_dir / "configs/full.yml").is_file():
            # Older prepared runs predate the suite-level snapshot of this parameter.
            residual["ppo"]["initial_zero_bias"] = read_yaml(self.run_dir / "configs/full.yml")["ppo"]["initial_zero_bias"]
        residual["high_level"] = dict(config=str(self.baseline_config), checkpoint=str(self.high_checkpoint))
        residual["data"].update(orders=str(self.orders_dir), parquet_root=str(self.data_dir / "parquet"))
        residual["runtime"].update(seed=self.s["seed"], device=self.s["device"],
            concurrency=self.s["residual_concurrency"], **execution,
            **{key: self.s[key] for key in RESIDUAL_EXECUTION_DEFAULTS})
        residual["training"].update(iterations=self.s["residual_iterations"],
            episodes_per_collect=self.s["episodes_per_collect"], buffer_size=self.s["residual_buffer_size"],
            batch_size=self.s["residual_batch_size"], repeat=self.s["residual_repeat"],
            validate_every=self.s["validate_every"], train_limit=None, valid_limit=None, test_limit=None)
        self.regime_path = self.run_dir / "artifacts/regime_full.npz"
        residual["regime"].update(model_path=str(self.regime_path), max_sequences=self.s["regime_max_sequences"],
                                   max_iter=self.s["regime_max_iter"])
        for variant in self.s["variants"]:
            cfg = copy.deepcopy(residual)
            for module in VARIANT_MODULES["full"]:
                cfg[module]["enabled"] = module in VARIANT_MODULES[variant]
            cfg["output"] = str(self.run_dir / variant)
            write_config(self.run_dir / "configs" / f"{variant}.yml", cfg)

    def check_baseline_source(self, orders):
        source = Path(self.s["baseline_source_run"])
        if json.loads((source / "orders_manifest.json").read_text()) != orders:
            raise ValueError("The reference PPO used different train, validation or test orders")
        for name, output in (("baseline_train", "baseline"), ("baseline_test", "baseline_test")):
            filename = Path("configs") / f"{name}.yml"
            expected = read_yaml(self.run_dir / filename)
            expected["trainer"]["checkpoint_path"] = str(source / output)
            if name == "baseline_test":
                expected["policy"]["kwargs"]["weight_file"] = str(source / "baseline/best_validation.pth")
            if read_yaml(source / filename) != expected:
                raise ValueError(f"Reference PPO settings differ from this run: {source / filename}")

    def copy_baseline_outputs(self, log_path, outputs):
        source = Path(self.s["baseline_source_run"])
        recorded = {}
        for marker in (source / "stages").glob("*.json"):
            recorded.update(json.loads(marker.read_text())["outputs"])
        imported = {}
        for dest in outputs:
            src = source / dest.relative_to(self.run_dir)
            dest.parent.mkdir(exist_ok=True)
            temp = dest.with_name(dest.name + ".tmp")
            shutil.copyfile(src, temp)
            digest = fingerprint(temp)
            if recorded.get(str(src)) != digest:
                raise RuntimeError(f"Reference output does not match a completed stage: {src}")
            temp.replace(dest)
            imported[str(src)] = digest
        selection = json.loads((self.run_dir / "baseline/selection.json").read_text())
        if fingerprint(self.high_checkpoint) != selection["sha256"]:
            raise RuntimeError("The frozen reference weights differ from its validation selection")
        self.check_result(self.run_dir / "baseline_test/result.csv")
        write_json(log_path, imported)
        self.emit(f"[Reused baseline] {', '.join(imported)}")

    def command(self, args, repo, log_path):
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONUNBUFFERED="1",
                   QLIB_BASELINE_REPO=str(self.baseline_repo), QLIB_RUNTIME_REPO=str(self.residual_repo))
        cmd = [str(ROOT / "envs/qlib/bin/python"), "-u", *map(str, args)]
        self.emit("[Command] " + shlex.join(cmd))
        with log_path.open("a", buffering=1) as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    self.emit(line, end="")
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            finally:
                process.stdout.close()

    def stage(self, name, action, outputs, partial=None):
        self.sequence += 1
        marker = self.run_dir / "stages" / f"{self.sequence:02d}_{name}.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            for filename, digest in record["outputs"].items():
                path = Path(filename)
                if not path.is_file() or fingerprint(path) != digest:
                    raise RuntimeError(f"Completed stage artifact missing/changed: {path}")
            self.emit(f"[Skip completed] {name}")
            return
        if partial is not None and partial.exists():
            if not self.restart:
                raise RuntimeError(f"Incomplete stage {name}: {partial}. Use --resume {self.run_dir} "
                                   "--restart-incomplete to archive this stage and restart it.")
            archive = self.run_dir / "interrupted" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive.mkdir(parents=True)
            shutil.move(str(partial), str(archive / partial.name))
            self.emit(f"[Archived incomplete output] {archive / partial.name}")
        self.emit(f"\n[Stage {self.sequence}] {name}")
        action(self.run_dir / "logs" / f"{self.sequence:02d}_{name}.log")
        for path in outputs:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Stage {name} did not produce {path}")
        write_json(marker, dict(name=name, completed=datetime.now().isoformat(),
                               outputs={str(path): fingerprint(path) for path in outputs}))
        self.emit(f"[Stage complete] {name}")

    def select_baseline(self, log_path):
        import pandas as pd
        import torch

        table = pd.read_csv(self.run_dir / "baseline/validation_result.csv")
        expected = self.s["baseline_iterations"] // self.s["validate_every"]
        if len(table) != expected or not table["val/pa"].map(math.isfinite).all():
            raise RuntimeError("Baseline validation history is incomplete or non-finite")
        best_row = int(table["val/pa"].to_numpy().argmax())
        iteration = (best_row + 1) * self.s["validate_every"]
        checkpoint = self.run_dir / "baseline/checkpoints" / f"{iteration:03d}.pth"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pa = float(table.iloc[best_row]["val/pa"])
        if payload["current_iter"] != iteration or not math.isclose(float(payload["metrics"]["val/pa"]), pa, abs_tol=1e-10):
            raise RuntimeError("Validation score and baseline checkpoint iteration do not match")
        shutil.copyfile(checkpoint, self.high_checkpoint)
        record = dict(iteration=iteration, validation_pa=pa, source=str(checkpoint), sha256=fingerprint(self.high_checkpoint))
        write_json(self.run_dir / "baseline/selection.json", record)
        log_path.write_text(json.dumps(record, indent=2))
        self.emit(f"[Frozen high-level PPO] iteration={iteration}, validation PA={pa:.6f}")

    def check_result(self, path):
        import pandas as pd

        table = pd.read_csv(path, dtype={"stock_id": "string"})
        if len(table) != self.test_count or table.duplicated(["stock_id", "datetime"]).any():
            raise RuntimeError(f"Incomplete or duplicated test orders in {path}")
        if not all(table[key].map(math.isfinite).all() for key in ("pa", "ffr")):
            raise RuntimeError(f"Non-finite test PA/FFR in {path}")
        return table

    def summarize(self, log_path):
        import pandas as pd

        rows, baseline_keys = [], None
        for variant in ["baseline", *self.s["variants"]]:
            path = self.run_dir / f"{variant}_test/result.csv"
            table = self.check_result(path)
            keys = set(zip(table.stock_id, table.datetime))
            if baseline_keys is None:
                baseline_keys = keys
            elif keys != baseline_keys:
                raise RuntimeError(f"Test order identities differ for {variant}")
            rows.append(dict(model=variant, orders=len(table), pa=float(table.pa.mean()), ffr=float(table.ffr.mean())))
        summary = pd.DataFrame(rows)
        summary.to_csv(self.run_dir / "summary.csv", index=False)
        log_path.write_text(summary.to_string(index=False))
        self.emit(summary.to_string(index=False))

    def report_execution_metrics(self, log_path=None):
        from execution_metrics import ablation_report, execution_report, order_references

        orders, digests = read_order_split(self.orders_dir / "test")
        manifest = self.run_dir / "orders_manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text())["test"]["files"] != digests:
            raise RuntimeError("Test order files differ from the saved experiment")
        self.test_count = len(orders)

        baseline = read_yaml(self.run_dir / "configs/baseline_test.yml")
        source = baseline["data"]["source"]
        minutes = baseline["simulator"].get("data_granularity", 1)
        start = source["default_start_time_index"] // minutes
        end = source["default_end_time_index"] // minutes
        base_reward_modes = {"baseline": baseline["reward"]["class"]} if self.progressive else {}
        for variant in self.s["variants"]:
            config = self.run_dir / f"{variant}_test/resolved_config.yml"
            if not config.is_file():
                config = self.run_dir / "configs" / f"{variant}.yml"
            resolved = read_yaml(config)
            if self.progressive:
                base_reward_modes[variant] = resolved.get("base_reward", "ppo_discrete")
                if base_reward_modes[variant] != self.s.get("residual_base_reward", "ppo_discrete"):
                    raise ValueError(f"Low-level reward for {variant} differs from the saved suite configuration")
            data = resolved["data"]
            if (data["start_index"], data["end_index"]) != (start, end):
                raise ValueError(f"Execution window for {variant} differs from baseline")

        results = {name: self.check_result(self.run_dir / f"{name}_test/result.csv")
                   for name in ["baseline", *self.s["variants"]]}
        references = order_references(orders, self.data_dir / "parquet/backtest", start, end)
        summary, details = execution_report(references, results)
        reports = {"order_execution_metrics.csv": details, "execution_metrics.csv": summary}
        if self.progressive:
            summary = summary.set_index("model").loc[["baseline", *PROGRESSIVE_VARIANTS]].reset_index()
            enabled = [VARIANT_MODULES.get(name, ()) for name in summary["model"]]
            summary.insert(1, "module_count", [len(modules) for modules in enabled])
            summary.insert(2, "enabled_modules", [" + ".join(modules) or "none" for modules in enabled])
            if "residual_base_reward" in self.s:
                summary.insert(3, "base_reward", summary["model"].map(base_reward_modes))
            reports["summary.csv"] = summary
        display = [summary.to_string(index=False)]
        if self.all_ablations:
            contrasts = ablation_report(details, LEAVE_ONE_OUT_VARIANTS)
            reports.update({"summary.csv": summary, "ablation_summary.csv": contrasts})
            display.extend(["[Module effects] full minus each ablation | PA and FFR: higher is better; "
                            "arrival cost and VWAP slippage: lower is better | FFR in percentage points",
                            contrasts.round(6).replace(-0.0, 0.0).to_string(index=False, float_format="{:.6f}".format),
                            f"[Ablation results] {self.run_dir / 'ablation_summary.csv'}"])
        for name, table in reports.items():
            table.to_csv(self.run_dir / (name + ".tmp"), index=False)
        for name in reports:
            (self.run_dir / (name + ".tmp")).replace(self.run_dir / name)
        display.append(f"[Execution metrics] {self.run_dir / 'execution_metrics.csv'}")
        report = "\n".join(display)
        self.emit(report)
        if log_path is not None:
            log_path.write_text(report + "\n")

    def execute(self):
        baseline_entry = ROOT / "scripts/original_qlib_baseline.py"
        selected = [self.high_checkpoint, self.run_dir / "baseline/selection.json"]
        baseline_result = self.run_dir / "baseline_test/result.csv"
        if self.s.get("baseline_source_run"):
            reference = [*selected, baseline_result]
            self.stage("reuse_baseline", lambda log: self.copy_baseline_outputs(log, reference),
                       reference, partial=self.run_dir / "baseline")
        else:
            self.stage("baseline_train", lambda log: self.command(
                [baseline_entry, "--config_path", self.baseline_config], self.baseline_repo, log),
                [self.run_dir / "baseline/train_result.csv", self.run_dir / "baseline/validation_result.csv",
                 self.run_dir / "baseline/checkpoints/latest.pth"], partial=self.run_dir / "baseline")
            self.stage("baseline_select_best", self.select_baseline, selected)
        if self.s["variants"]:
            self.stage("fit_regime", lambda log: self.command(
                ["-m", "qlib.rl.contrib.train_residual", "fit-regime", "--config", self.run_dir / "configs/full.yml"],
                self.residual_repo, log), [self.regime_path], partial=self.regime_path)
        for variant in self.s["variants"]:
            if variant == "no_residual":
                continue  # Uniform execution uses the frozen high policy and has no low policy to train.
            self.stage(f"train_{variant}", lambda log, name=variant: self.command(
                ["-m", "qlib.rl.contrib.train_residual", "train", "--config", self.run_dir / "configs" / f"{name}.yml"],
                self.residual_repo, log),
                [self.run_dir / variant / "best_validation.pth", self.run_dir / variant / "train_result.csv",
                 self.run_dir / variant / "validation_result.csv", self.run_dir / variant / "resolved_config.yml"],
                partial=self.run_dir / variant)

        def evaluate_baseline(log):
            # The original CsvWriter requires its parent experiment directory.
            (self.run_dir / "baseline_test").mkdir()
            self.command([baseline_entry, "--config_path", self.run_dir / "configs/baseline_test.yml",
                          "--no_training", "--run_backtest"], self.baseline_repo, log)
            self.check_result(self.run_dir / "baseline_test/result.csv")

        if not self.s.get("baseline_source_run"):
            self.stage("test_baseline", evaluate_baseline, [baseline_result], partial=self.run_dir / "baseline_test")
        for variant in self.s["variants"]:
            def evaluate_variant(log, name=variant):
                if name == "no_residual":
                    model_args = ["--config", self.run_dir / "configs/no_residual.yml"]
                else:
                    model_args = ["--config", self.run_dir / name / "resolved_config.yml",
                                  "--checkpoint", self.run_dir / name / "best_validation.pth"]
                self.command(["-m", "qlib.rl.contrib.train_residual", "evaluate", *model_args,
                              "--split", "test", "--set", f"output={self.run_dir / (name + '_test')}"],
                             self.residual_repo, log)
                self.check_result(self.run_dir / f"{name}_test/result.csv")
            self.stage(f"test_{variant}", evaluate_variant, [self.run_dir / f"{variant}_test/result.csv"],
                       partial=self.run_dir / f"{variant}_test")
        if self.progressive or self.all_ablations:
            reports = ["summary.csv", "execution_metrics.csv", "order_execution_metrics.csv"]
            if self.all_ablations:
                reports.append("ablation_summary.csv")
            self.stage("summary", self.report_execution_metrics, [self.run_dir / name for name in reports])
            final_report = "summary.csv" if self.progressive else "ablation_summary.csv"
        else:
            self.stage("summary", self.summarize, [self.run_dir / "summary.csv"])
            self.report_execution_metrics()
            final_report = "execution_metrics.csv"
        self.emit(f"\n[All complete] {self.run_dir / final_report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Suite settings; defaults to configs/full_experiment.yml")
    parser.add_argument("--run-dir", type=Path, help="New output directory; default is a timestamped directory")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="Set CPU sampling workers for models trained or evaluated in a new run")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline-only", action="store_true",
                       help="Train, select and evaluate only the original Qlib PPO in a new run")
    group.add_argument("--progressive", action="store_true",
                       help="Train a new original PPO and all three progressive variants in one run")
    group.add_argument("--ablation-from", type=Path, metavar="BASELINE_RUN",
                       help="Run all three leave-one-out ablations using this run's frozen PPO and order splits")
    group.add_argument("--progressive-from", type=Path, metavar="BASELINE_RUN",
                       help="Add residual PPO, market regime features, then reward shaping to a fixed baseline")
    parser.add_argument("--resume", type=Path, help="Skip completed stages using this run's saved settings")
    parser.add_argument("--metrics-only", type=Path, metavar="RUN_DIR",
                        help="Compute execution reports from saved test results and market data")
    parser.add_argument("--restart-incomplete", action="store_true", help="Archive and restart an incomplete stage")
    parser.add_argument("--prepare-only", action="store_true", help="Write configs/manifests without training or testing")
    args = parser.parse_args()
    baseline_source = args.progressive_from or args.ablation_from
    if (args.workers is not None or args.baseline_only or args.progressive or baseline_source) and (args.resume or args.metrics_only):
        parser.error("Model and worker options apply to new runs only")
    if baseline_source and args.config:
        parser.error("Experiments using an existing baseline use its saved configuration")
    if args.metrics_only and (args.resume or args.run_dir or args.config or args.prepare_only or args.restart_incomplete):
        parser.error("--metrics-only takes an existing experiment directory and runs on its own")
    if args.resume and (args.config or args.run_dir):
        parser.error("--resume uses its saved settings and cannot be combined with --config / --run-dir")
    if args.restart_incomplete and not args.resume:
        parser.error("--restart-incomplete requires --resume")
    if args.resume or args.metrics_only:
        run_dir = (args.resume or args.metrics_only).resolve()
        settings = read_yaml(run_dir / "suite_settings.yml")
    else:
        if baseline_source:
            source = baseline_source.resolve()
            settings = read_yaml(source / "suite_settings.yml")
            variants = list(PROGRESSIVE_VARIANTS) if args.progressive_from else ["full", *LEAVE_ONE_OUT_VARIANTS]
            settings.update(variants=variants, baseline_source_run=str(source))
            settings["residual_concurrency"] = args.workers if args.workers is not None else 16
        else:
            settings = read_yaml(args.config or ROOT / "configs/full_experiment.yml")
        if args.workers is not None and not baseline_source:
            settings.update(baseline_concurrency=args.workers, residual_concurrency=args.workers)
        if args.baseline_only:
            settings["variants"] = []
        elif args.progressive:
            settings.pop("baseline_source_run", None)
            settings["variants"] = list(PROGRESSIVE_VARIANTS)
        if settings["variants"]:
            template_dir = ROOT / settings["residual_repo"] / "examples/rl_order_execution/exp_configs"
            template = read_yaml(template_dir / "train_residual.yml")
            settings.setdefault("initial_zero_bias", template["ppo"]["initial_zero_bias"])
            settings.setdefault("residual_base_reward", template.get("base_reward", "ppo_discrete"))
        prefix = "iteration_" if "iteration_stocks" in settings else "full_suite_"
        if set(settings["variants"]) == set(PROGRESSIVE_VARIANTS):
            prefix += "progressive_"
        elif set(LEAVE_ONE_OUT_VARIANTS) <= set(settings["variants"]):
            prefix += "ablation_"
        elif not settings["variants"]:
            prefix += "baseline_"
        run_dir = (args.run_dir or ROOT / "results" / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if not args.metrics_only:
        validate_settings(settings)
    if not (args.resume or args.metrics_only):
        run_dir.mkdir(parents=True, exist_ok=False)
        write_config(run_dir / "suite_settings.yml", settings)
    with (run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This experiment is already running") from exc
        suite = Suite(run_dir, settings, args.restart_incomplete)
        try:
            suite.emit(f"[Run directory] {run_dir}")
            if args.metrics_only:
                suite.report_execution_metrics()
                return
            suite.prepare()
            if args.prepare_only:
                suite.emit(f"[Prepared only] Start with: bash {ROOT / 'run_full_experiment.sh'} --resume {run_dir}")
                return
            suite.execute()
        finally:
            suite.pipeline_log.close()


if __name__ == "__main__":
    def stop_handler(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("[Stopped] Completed stages are retained; use --resume with this run directory.", flush=True)
        sys.exit(130)
