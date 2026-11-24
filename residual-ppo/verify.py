"""Check the recorded source, artifacts and execution metrics using the standard library."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
SERVER_ROOT = PurePosixPath("/root/autodl-tmp/qlib-work")
RUNS = {
    "arrival_bps": "iteration_progressive_20261114_205706_552014",
    "price_shaping_v1": "iteration_full_price_shaping_v1",
    "paired_pa_v2": "iteration_full_paired_pa_v2",
}
METRICS = ("pa", "ffr", "arrival_is_bps", "vwap_slippage_bps")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_files(version, manifest):
    for original, expected in manifest.items():
        relative = PurePosixPath(original).relative_to(SERVER_ROOT)
        if relative.parts[0] == "results":
            file = ROOT / relative
        else:
            file = ROOT / "versions" / version / relative
            if not file.is_file():
                file = ROOT / "common" / relative
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{version}: SHA-256 mismatch for {relative}")


def verify_metrics(run, reference_orders):
    by_model = defaultdict(list)
    for row in read_table(run / "order_execution_metrics.csv"):
        by_model[row["model"]].append(row)
    summary = read_table(run / "summary.csv")
    if set(by_model) != {row["model"] for row in summary}:
        raise ValueError(f"{run.name}: model lists differ")
    for row in summary:
        model = row["model"]
        orders = by_model[model]
        order_ids = {(o["stock_id"], o["date"]) for o in orders}
        if len(order_ids) != len(orders) or len(orders) != int(row["orders"]):
            raise ValueError(f"{run.name}/{model}: order count or uniqueness differs")
        if reference_orders is None:
            reference_orders = order_ids
        if order_ids != reference_orders:
            raise ValueError(f"{run.name}/{model}: development orders differ")
        for metric in METRICS:
            values = [float(o[metric]) for o in orders if o[metric]]
            if metric == "vwap_slippage_bps" and len(values) != int(row["vwap_orders"]):
                raise ValueError(f"{run.name}/{model}: VWAP order count differs")
            mean = math.fsum(values) / len(values)
            if not math.isclose(mean, float(row[metric]), rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f"{run.name}/{model}: {metric} differs")
    return reference_orders


def main():
    reference_orders = None
    reference_split = None
    for version, directory in RUNS.items():
        run = ROOT / "results" / directory
        sources = read_json(run / "sources_initial.json")
        if sources != read_json(run / "sources_latest.json"):
            raise ValueError(f"{version}: source manifests differ")
        verify_files(version, sources)
        stages = sorted((run / "stages").glob("*.json"))
        for stage in stages:
            verify_files(version, read_json(stage)["outputs"])
        split = read_json(run / "orders_manifest.json")["test"]
        if reference_split is None:
            reference_split = split
        if split != reference_split:
            raise ValueError(f"{version}: test split differs")
        reference_orders = verify_metrics(run, reference_orders)
        if len(reference_orders) != split["count"]:
            raise ValueError(f"{version}: test manifest count differs from the reports")
        print(f"{version}: {len(sources)} source hashes, {len(stages)} stages, "
              f"{len(reference_orders)} matched orders and four metrics verified")


if __name__ == "__main__":
    main()
