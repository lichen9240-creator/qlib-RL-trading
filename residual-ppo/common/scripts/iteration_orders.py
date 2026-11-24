"""Build a reproducible development order set without touching the final test period."""
import argparse
import errno
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def build_orders(train, validation, stock_count, seed):
    frames = {}
    for name, source in (("train", train), ("valid", validation)):
        rows = source.copy()
        rows["date"] = pd.to_datetime(rows["date"])
        if rows[["instrument", "date"]].isna().any().any() or rows.duplicated(["instrument", "date"]).any():
            raise ValueError(f"Expected one order per stock and date in {name}")
        frames[name] = rows
    train, validation = frames["train"], frames["valid"]
    if train.empty or validation.empty or train["date"].max() >= validation["date"].min():
        raise ValueError("Source training and validation periods must be chronological and disjoint")

    # Stock selection uses identifiers from the training period only.
    universe = sorted(train["instrument"].unique(),
                      key=lambda stock: (sha256(f"{seed}:{stock}".encode()).digest(), stock))
    if not 1 <= stock_count <= len(universe):
        raise ValueError(f"Requested {stock_count} of {len(universe)} available training stocks")
    stocks = sorted(universe[:stock_count])
    train = train.loc[train["instrument"].isin(stocks)]
    validation = validation.loc[validation["instrument"].isin(stocks)]
    dates = sorted(validation["date"].unique())
    if len(dates) < 2:
        raise ValueError("Need at least two validation dates to form development splits")
    cutoff = dates[len(dates) // 2]
    splits = {"train": train, "valid": validation.loc[validation["date"] < cutoff],
              "test": validation.loc[validation["date"] >= cutoff]}
    tables = {}
    for name, rows in splits.items():
        if set(rows["instrument"]) != set(stocks):
            raise ValueError(f"Some training-universe stocks have no orders in {name}")
        tables[name] = rows.set_index(["date", "instrument"]).sort_index()
    return stocks, tables


def prepare(source, destination, stock_count, seed):
    from full_experiment import fingerprint, read_order_split

    if destination.exists():
        manifest = json.loads((destination / "selection.json").read_text())
        if seed != manifest["seed"] or stock_count != len(manifest["stocks"]):
            raise ValueError(f"Different stock selection already exists at {destination}")
        for name in ("train", "valid"):
            current = {p.name: fingerprint(p) for p in (source / name).iterdir() if p.is_file()}
            if current != manifest["source_files"][name]:
                raise ValueError(f"Source {name} orders changed since {destination} was prepared")
        for name in ("train", "valid", "test"):
            folder = destination / name
            if (sorted(p.name for p in folder.iterdir()) != ["orders.pkl"]
                    or fingerprint(folder / "orders.pkl") != manifest["splits"][name]["sha256"]):
                raise ValueError(f"Saved {name} orders differ from their selection manifest")
        print(f"[Iteration orders] {destination}", flush=True)
        return

    train, train_files = read_order_split(source / "train")
    validation, valid_files = read_order_split(source / "valid")
    stocks, tables = build_orders(train, validation, stock_count, seed)
    manifest = {"seed": seed, "stocks": stocks, "source_files": {"train": train_files, "valid": valid_files},
                "splits": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".iteration-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "orders"
        for name, table in tables.items():
            directory = staging / name
            directory.mkdir(parents=True)
            path = directory / "orders.pkl"
            table.to_pickle(path, protocol=4)
            dates = table.index.get_level_values("date")
            manifest["splits"][name] = dict(orders=len(table), stocks=table.index.unique("instrument").size,
                                            start=str(dates.min()), end=str(dates.max()), sha256=fingerprint(path))
        (staging / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        try:
            staging.rename(destination)
        except OSError as exc:  # Another run published this shared set first.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if json.loads((destination / "selection.json").read_text()) != manifest:
                raise ValueError(f"Concurrent stock selections differ at {destination}") from exc
    print(json.dumps(manifest["splits"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from full_experiment import ROOT, read_yaml

    config = read_yaml(args.config)
    prepare(ROOT / config["data_dir"] / "orders", ROOT / config["orders_dir"],
            config["iteration_stocks"], config["seed"])
