"""Test-set transaction cost analysis for the Qlib Simple execution experiment."""
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["stock_id", "date"]
BPS = 10_000


def order_references(orders, backtest_dir, start, end):
    """Build a common arrival, deadline and market VWAP benchmark for each parent order."""
    if not 0 <= start < end:
        raise ValueError(f"Invalid execution window [{start}, {end})")
    parents = orders[["instrument", "date", "order_type", "amount"]].rename(
        columns={"instrument": "stock_id", "amount": "order_amount"}
    ).copy()
    parents["stock_id"] = parents["stock_id"].astype("string")
    parents["date"] = pd.to_datetime(parents["date"]).dt.normalize()
    if parents.empty or parents[KEY].isna().any().any() or parents.duplicated(KEY).any():
        raise ValueError("Expected one test order per stock and date")
    if not parents["order_type"].isin([0, 1]).all():
        raise ValueError("Expected Qlib order_type 0 (sell) or 1 (buy)")
    if not (np.isfinite(parents["order_amount"]) & (parents["order_amount"] > 0)).all():
        raise ValueError("Invalid parent order amount")
    parents["side"] = parents.pop("order_type").map({0: "sell", 1: "buy"})

    benchmarks = []
    for stock, stock_orders in parents.groupby("stock_id", sort=False):
        if Path(stock).name != stock:
            raise ValueError(f"Invalid stock identifier: {stock}")
        path = backtest_dir / f"{stock}.parquet"
        bars = pd.read_parquet(path, columns=["$close0", "$volume0"]).xs(stock, level="instrument")
        dates = bars.index.normalize()
        bars = bars.loc[dates.isin(stock_orders["date"])]
        if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
            raise ValueError(f"Unordered or duplicated market bars for {stock}")
        dates = bars.index.normalize()
        offsets = bars.groupby(dates).cumcount()
        window = bars.loc[(offsets >= start) & (offsets < end)].astype("float64")
        by_date = window.groupby(window.index.normalize())
        counts = by_date.size().reindex(stock_orders["date"], fill_value=0)
        if (counts != end - start).any():
            raise ValueError(f"Incomplete execution window for {stock} on {counts[counts != end - start].index[0]}")
        if (not np.isfinite(window.to_numpy()).all() or (window["$close0"] <= 0).any()
                or (window["$volume0"] < 0).any()):
            raise ValueError(f"Invalid execution-window price or volume for {stock}")
        prices = by_date["$close0"]
        refs = pd.DataFrame({"arrival_price": prices.first(), "deadline_price": prices.last()})
        volume = by_date["$volume0"].sum()
        notional = (window["$close0"] * window["$volume0"]).groupby(window.index.normalize()).sum()
        refs["market_vwap"] = notional / volume.where(volume > 0)
        refs.index.name = "date"
        refs["stock_id"] = stock
        benchmarks.append(refs.reset_index())
    prices = pd.concat(benchmarks, ignore_index=True).set_index(KEY)
    return parents.set_index(KEY).join(prices, validate="one_to_one").sort_index()


def execution_report(references, results):
    """Return order-weighted summaries and the matched order-level inputs and costs."""
    summaries, details = [], []
    for model, raw in results.items():
        fills = raw[["stock_id", "datetime", "pa", "ffr", "trade_price"]].copy()
        fills["stock_id"] = fills["stock_id"].astype("string")
        fills["date"] = pd.to_datetime(fills["datetime"]).dt.normalize()
        fills = fills.set_index(KEY).sort_index()
        if fills.index.has_duplicates or not fills.index.equals(references.index):
            raise ValueError(f"Test order identities differ from parent orders for {model}")
        detail = references.join(fills, validate="one_to_one")
        ffr = detail["ffr"]
        if not np.isfinite(detail["pa"]).all() or not ffr.between(-1e-8, 1 + 1e-8).all():
            raise ValueError(f"Invalid PA or fill fraction for {model}")
        filled = ffr.clip(0, 1)  # Floating-point fills can exceed the parent by roundoff.
        price = detail["trade_price"].where(filled > 0)
        if not (np.isfinite(price[filled > 0]) & (price[filled > 0] > 0)).all():
            raise ValueError(f"Missing or invalid execution price on a filled order for {model}")
        sign = detail["side"].map({"buy": 1, "sell": -1})
        arrival = detail["arrival_price"]
        # Mark any remaining inventory at the deadline, including entirely unfilled orders.
        executed_cost = filled * (price.fillna(arrival) - arrival)
        opportunity_cost = (1 - filled) * (detail["deadline_price"] - arrival)
        shortfall = BPS * sign * (executed_cost + opportunity_cost) / arrival
        slippage = BPS * sign * (price / detail["market_vwap"] - 1)
        detail["arrival_is_bps"] = shortfall
        detail["vwap_slippage_bps"] = slippage
        detail.insert(0, "model", model)
        details.append(detail.reset_index())

        summaries.append(dict(
            model=model, orders=len(detail), pa=detail["pa"].mean(), ffr=ffr.mean(),
            arrival_is_bps=shortfall.mean(), vwap_slippage_bps=slippage.mean(),
            vwap_orders=int(slippage.count()),
        ))
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def ablation_report(details, variants):
    """Matched full-minus-ablation changes. VWAP uses fills observed in both arms."""
    def orders_for(model):
        return details.loc[details["model"] == model].set_index(KEY, verify_integrity=True).sort_index()

    full = orders_for("full")
    rows = []
    for name in variants:
        other = orders_for(name)
        if full.empty or not full.index.equals(other.index):
            raise ValueError(f"Different evaluation orders for full and {name}")
        pa = full["pa"] - other["pa"]
        slippage = (full["vwap_slippage_bps"] - other["vwap_slippage_bps"]).dropna()
        rows.append(dict(
            ablated_model=name, orders=len(pa), delta_pa_bps=pa.mean(),
            delta_ffr_pp=(full["ffr"] - other["ffr"]).mean() * 100,
            delta_arrival_is_bps=(full["arrival_is_bps"] - other["arrival_is_bps"]).mean(),
            delta_vwap_slippage_bps=slippage.mean(), paired_vwap_orders=len(slippage),
        ))
    return pd.DataFrame(rows)
