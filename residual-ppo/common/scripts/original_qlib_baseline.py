"""Original Qlib algorithm with cached data IO and persistent CPU environments.

The wrapper configures parallel environments and replay buffer capacity. The
original policy, reward, simulator, Collector and train/validation loops are retained.
The original repository is never edited by this wrapper.
"""
import argparse
from functools import partial
import logging
import os
from pathlib import Path
import sys
import warnings

import qlib
import qlib.rl.utils

expected = Path(os.environ["QLIB_BASELINE_REPO"]).resolve()
if Path(qlib.__file__).resolve().parent != expected / "qlib":
    raise RuntimeError(f"Wrong Qlib import for original baseline: {qlib.__file__}")

# Spawn re-imports this guarded wrapper before unpickling the worker target.
# Expose ONLY the additional runtime utility, keeping every original RL module
# resolved from QLIB_BASELINE_REPO. Do not run training while importing.
runtime_repo = Path(os.environ.get("QLIB_RUNTIME_REPO", str(expected.parent / "qlib-modules"))).resolve()
qlib.rl.utils.__path__.append(str(runtime_repo / "qlib/rl/utils"))

from parquet_adapter import install
from qlib.rl.utils.parallel_runtime import ParallelExecutionTrainer

# Multiprocessing finalizers may close handles before DataQueue.__del__ runs.
# Keep this process-only compatibility fix out of the original repository.
from qlib.rl.utils.data_queue import DataQueue

original_destructor = DataQueue.__del__


def safe_destructor(self):
    if not getattr(self, "_activated", False) or sys.is_finalizing():
        return
    try:
        original_destructor(self)
    except (OSError, ValueError) as exc:
        if "closed" not in str(exc).lower():
            raise


DataQueue.__del__ = safe_destructor


def main():
    from ruamel.yaml import YAML
    import qlib.rl.trainer.api as trainer_api
    from qlib.rl.contrib.train_onpolicy import main as original_main

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--no_training", action="store_true")
    parser.add_argument("--run_backtest", action="store_true")
    args = parser.parse_args()
    with open(args.config_path) as stream:
        config = YAML(typ="safe", pure=True).load(stream)
    runtime = config["env"]
    initialize_data = partial(install, file_cache_size=runtime.get("file_cache_size", 256),
                              day_cache_size=runtime.get("day_cache_size", 4096))
    initialize_data()
    trainer_api.Trainer = partial(ParallelExecutionTrainer,
        execution_mode=runtime.get("parallel_mode", "subproc"),
        worker_threads=runtime.get("worker_threads", 1),
        worker_timeout=runtime.get("worker_timeout", 120), worker_initializer=initialize_data)
    # train_onpolicy forwards episode and update settings to TrainingVessel;
    # supply its buffer capacity here for larger concurrent collects.
    if not args.no_training and "buffer_size" in config["trainer"]:
        trainer_api.TrainingVessel = partial(trainer_api.TrainingVessel, buffer_size=config["trainer"]["buffer_size"])
    warnings.filterwarnings("ignore", message=r"np\.find_common_type is deprecated\.",
                            category=DeprecationWarning, module=r"pandas\.core\.dtypes\.cast")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(f"[Original Qlib] Repository: {expected}", flush=True)
    print(f"[Runtime] {runtime.get('parallel_mode', 'subproc')}, {runtime['concurrency']} environments", flush=True)
    original_main(config, run_training=not args.no_training, run_backtest=args.run_backtest)


if __name__ == "__main__":
    main()
