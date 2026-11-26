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


RUN = Path(__file__).resolve().parent
EXAMPLE = RUN.parents[1] / "example"
CONFIG = RUN / "train_opds.yml"
STATUS = RUN / "status.json"
STARTED = perf_counter()
status = {"pid": os.getpid(), "completed_iterations": 0}


def update_status(**changes):
    status.update(changes)
    status["updated_at"] = datetime.now().astimezone().isoformat()
    status["elapsed_seconds"] = round(perf_counter() - STARTED, 2)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


class Progress(Callback):
    def on_fit_start(self, trainer, vessel):
        device = next(vessel.policy.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Expected a CUDA policy, got {device}")
        update_status(phase="training", device=str(device), gpu=torch.cuda.get_device_name(device))
        print(f"Training {vessel.policy.__class__.__name__} with OPDS reward on {status['gpu']} ({device})", flush=True)

    def on_iter_start(self, trainer, vessel):
        self.started = perf_counter()
        update_status(current_iteration=trainer.current_iter + 1)

    def on_iter_end(self, trainer, vessel):
        update_status(
            completed_iterations=trainer.current_iter,
            training_episodes=trainer.current_episode,
            last_iteration_seconds=round(perf_counter() - self.started, 2),
            peak_tensor_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
        )

    def on_fit_end(self, trainer, vessel):
        update_status(phase="testing", training_seconds=round(perf_counter() - STARTED, 2))


def main():
    os.chdir(EXAMPLE)
    set_log_with_config(C.logging_config)
    torch.set_num_threads(2)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    update_status(phase="starting", max_iterations=config["trainer"]["max_epoch"])
    original_train = train_onpolicy.train

    def train_with_progress(**kwargs):
        kwargs["trainer_kwargs"]["callbacks"].append(Progress())
        return original_train(**kwargs)

    train_onpolicy.train = train_with_progress
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(0x80000001)
    try:
        train_onpolicy.main(config, run_training=True, run_backtest=True)
        update_status(phase="complete")
    except BaseException:
        update_status(phase="failed", error=traceback.format_exc())
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
