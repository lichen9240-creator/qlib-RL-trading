"""Train a fresh OPDS policy with the frozen official configuration."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

import torch
import yaml
from qlib.config import C
from qlib.log import set_log_with_config
from qlib.rl.contrib import train_onpolicy
from qlib.rl.trainer.callbacks import Callback


PROJECT = Path(__file__).resolve().parents[1]


class Progress(Callback):
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.started = perf_counter()
        self.status = {"pid": os.getpid(), "completed_iterations": 0}

    def write(self, **changes):
        self.status.update(changes)
        self.status["updated_at"] = datetime.now().astimezone().isoformat()
        self.status["elapsed_seconds"] = round(perf_counter() - self.started, 2)
        temporary = self.run_dir / "status.tmp"
        temporary.write_text(json.dumps(self.status, indent=2), encoding="utf-8")
        temporary.replace(self.run_dir / "status.json")

    def on_fit_start(self, trainer, vessel):
        device = next(vessel.policy.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Expected a CUDA policy, got {device}")
        # Save the exact initialization before the first collected episode.
        torch.save(vessel.policy.state_dict(), self.run_dir / "initial_model.pth")
        self.write(phase="training", device=str(device), gpu=torch.cuda.get_device_name(device))

    def on_iter_start(self, trainer, vessel):
        self.iteration_started = perf_counter()
        self.write(current_iteration=trainer.current_iter + 1)

    def on_iter_end(self, trainer, vessel):
        self.write(
            completed_iterations=trainer.current_iter,
            episode_counter_including_validation=trainer.current_episode,
            last_iteration_seconds=round(perf_counter() - self.iteration_started, 2),
            peak_tensor_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
        )

    def on_fit_end(self, trainer, vessel):
        self.write(phase="testing", training_seconds=round(perf_counter() - self.started, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/train_opds.yml")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir or PROJECT / "runs" / datetime.now().strftime("train_%Y%m%d_%H%M%S")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config["trainer"]["checkpoint_path"] = str(run_dir / "outputs")
    (run_dir / "train_opds.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    progress = Progress(run_dir)
    progress.write(phase="starting", max_iterations=config["trainer"]["max_epoch"])
    set_log_with_config(C.logging_config)
    torch.set_num_threads(2)
    original_train = train_onpolicy.train

    def train_with_progress(**kwargs):
        kwargs["trainer_kwargs"]["callbacks"].append(progress)
        return original_train(**kwargs)

    train_onpolicy.train = train_with_progress
    kernel = ctypes.windll.kernel32 if os.name == "nt" else None
    if kernel is not None:
        kernel.SetThreadExecutionState(0x80000001)
    print(f"Run directory: {run_dir}", flush=True)
    try:
        train_onpolicy.main(config, run_training=True, run_backtest=True)
        progress.write(phase="complete")
    except BaseException:
        progress.write(phase="failed", error=traceback.format_exc())
        raise
    finally:
        if kernel is not None:
            kernel.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
