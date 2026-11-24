"""User-invoked fixed-checkpoint parity check; never trains or changes artifacts."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--orders", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--reference", choices=("parallel", "serial"), default="parallel",
                        help="parallel isolates batching/GRU changes; serial also checks caches and multiprocessing")
    args = parser.parse_args()
    if args.orders < 1 or args.concurrency < 1:
        parser.error("orders and concurrency must be positive")
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    output = ROOT / "results" / ("runtime_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    tables = []
    env = dict(os.environ, PYTHONPATH=str(ROOT / "qlib-modules"), PYTHONUNBUFFERED="1")
    reference_mode = "subproc" if args.reference == "parallel" else "dummy"
    for name, mode, optimized in (("reference_" + args.reference, reference_mode, False),
                                   ("optimized_parallel", "subproc", True)):
        run = output / name
        command = [str(ROOT / "envs/qlib/bin/python"), "-u", "-m", "qlib.rl.contrib.train_residual",
                   "evaluate", "--config", str(args.config.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
                   "--split", "valid", "--set", f"output={run}", "--set", f"training.valid_limit={args.orders}",
                   "--set", f"runtime.concurrency={args.concurrency}", "--set", f"runtime.parallel_mode={mode}",
                   "--set", f"runtime.high_batch_size={args.concurrency if optimized else 1}",
                   "--set", f"runtime.gru_sequence={'true' if optimized else 'false'}",
                   "--set", "runtime.high_batch_wait_ms=0.5", "--set", "runtime.profile_execution=true"]
        sizes = dict(file_cache_size=256, day_cache_size=4096, regime_cache_size=4096)
        for key, value in sizes.items():
            command.extend(["--set", f"runtime.{key}={0 if mode == 'dummy' else value}"])
        print(f"[Check] {name}: same checkpoint, first {args.orders} validation orders", flush=True)
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
        table = pd.read_csv(run / "result.csv")
        if len(table) != args.orders or table.duplicated(["stock_id", "datetime"]).any():
            raise ValueError(f"Expected {args.orders} unique stock-day orders in {run}")
        tables.append(table.set_index(["stock_id", "datetime"]).sort_index())
    left, right = tables
    pd.testing.assert_index_equal(left.index, right.index)
    for key in ("pa", "ffr", "trade_price", "trade_value"):
        if key not in left or key not in right:
            raise ValueError(f"Missing unchanged execution metric: {key}")
        if not np.isfinite(left[key]).all() or not np.isfinite(right[key]).all():
            raise ValueError(f"Non-finite {key}")
        np.testing.assert_allclose(left[key], right[key], rtol=1e-10, atol=1e-7, err_msg=key)
    report = dict(passed=True, orders=len(left), checkpoint=str(args.checkpoint.resolve()),
                  reference_mode=args.reference,
                  reference={key: float(left[key].mean()) for key in ("pa", "ffr")},
                  optimized={key: float(right[key].mean()) for key in ("pa", "ffr")})
    (output / "parity.json").write_text(json.dumps(report, indent=2))
    print(f"[Parity passed] {output / 'parity.json'}", flush=True)


if __name__ == "__main__":
    main()
