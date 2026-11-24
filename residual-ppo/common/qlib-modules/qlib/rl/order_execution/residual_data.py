# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Data-only adapters; no DatasetH unpickling or global loader changes."""
from collections import OrderedDict
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir
from qlib.rl.data.base import BaseIntradayProcessedData, ProcessedDataProvider
from qlib.rl.data.native import DataframeIntradayBacktestData
from qlib.rl.order_execution.simulator_simple import SingleAssetOrderExecutionSimple
from qlib.utils.pickle_utils import restricted_pickle_load


class ParquetStore:
    """Bounded process-local caches; public day frames are defensive copies.

    Data files and the fitted regime model must remain fixed for one run.
    Spawned workers get empty caches, which then survive successive collects.
    """
    def __init__(self, root, file_cache_size=256, day_cache_size=4096, regime_cache_size=4096):
        self.root = Path(root).resolve()
        self.file_cache_size = file_cache_size
        self.day_cache_size = day_cache_size
        self.regime_cache_size = regime_cache_size
        if any(type(n) is not int or n < 0 for n in (file_cache_size, day_cache_size, regime_cache_size)):
            raise ValueError("Cache sizes must be nonnegative integers; zero disables caching")
        self._frames, self._days, self._regimes = OrderedDict(), OrderedDict(), OrderedDict()
        self._lock = threading.RLock()

    def __getstate__(self):
        return {key: getattr(self, key) for key in ("root", "file_cache_size", "day_cache_size", "regime_cache_size")}

    def __setstate__(self, state):
        self.__init__(**state)

    def _cached(self, cache, capacity, key, loader):
        if capacity == 0:
            return loader()
        with self._lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        value = loader()
        with self._lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > capacity:
                cache.popitem(last=False)
        return value

    def clear(self):
        with self._lock:
            self._frames.clear()
            self._days.clear()
            self._regimes.clear()

    def _day(self, kind, stock, date):
        path = (self.root / kind / f"{stock}.parquet").resolve()
        if kind not in ("feature", "backtest") or not path.is_relative_to(self.root / kind):
            raise ValueError("Invalid market-data path")
        start = pd.Timestamp(date).normalize()

        def load_day():
            frame = self._cached(self._frames, self.file_cache_size, str(path), lambda: pd.read_parquet(path))
            day = frame.loc[pd.IndexSlice[stock, start:start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)], :]
            day = day.droplevel("instrument").copy()
            if day.empty or not day.index.is_monotonic_increasing or day.index.has_duplicates:
                raise ValueError(f"Missing or invalid stock-day: {stock} {date}")
            return day

        return self._cached(self._days, self.day_cache_size, (kind, stock, start.value), load_day)

    def day(self, kind, stock, date):
        return self._day(kind, stock, date).copy()

    def regime_features(self, stock, date, regime, enabled):
        # Only cache functions of the market day and FROZEN model. Order budgets,
        # policy actions, and execution history must never enter this cache.
        from qlib.rl.order_execution.regime import causal_features

        key = (stock, pd.Timestamp(date).normalize().value, regime, bool(enabled))

        def calculate():
            raw = causal_features(self._day("backtest", stock, date))
            features = regime.normalize(raw).astype(np.float32)
            beliefs = regime.filter(raw) if enabled else np.zeros((len(raw), regime.n_states))
            features.setflags(write=False)
            beliefs.setflags(write=False)
            return features, beliefs

        return self._cached(self._regimes, self.regime_cache_size, key, calculate)


class ParquetProcessedDataProvider(ProcessedDataProvider):
    def __init__(self, store, today=None, yesterday=None):
        self.store = store
        self.today = today or ["$high", "$low", "$open", "$close", "$volume"]
        self.yesterday = yesterday or [f"{x}_1" for x in self.today]
        self._last_key, self._last_features = None, None

    def get_data(self, stock_id, date, feature_dim, time_index):
        # Each decision in an episode requests the same raw features. Keep only
        # one day per worker; the interpreter masks future bars on a private copy.
        key = (stock_id, pd.Timestamp(date), feature_dim, tuple(self.today), tuple(self.yesterday))
        features = self._last_features
        cached = self.store.day_cache_size > 0
        if not cached or key != self._last_key or not features.today.index.equals(time_index):
            frame = self.store.day("feature", stock_id, date)
            if not frame.index.equals(time_index):
                frame = frame.reindex(time_index)
            features = BaseIntradayProcessedData()
            features.today, features.yesterday = frame[self.today], frame[self.yesterday]
            if (features.today.shape[1] != feature_dim or features.today.isna().any().any()
                    or features.yesterday.isna().any().any()):
                raise ValueError("High-level features do not match the frozen baseline")
            if cached:
                self._last_key, self._last_features = key, features
        result = BaseIntradayProcessedData()
        result.today, result.yesterday = features.today.copy(), features.yesterday.copy()
        return result


class FrameSimulator(SingleAssetOrderExecutionSimple):
    def __init__(self, order, frame, ticks_per_step=30, data_granularity=5):
        self.frame = frame
        super().__init__(order, Path("."), data_granularity=data_granularity, ticks_per_step=ticks_per_step)

    def get_backtest_data(self):
        return DataframeIntradayBacktestData(self.frame)


class ParquetOrderDataset:
    """Original example's order format and time-index semantics, lazy market reads."""
    def __init__(self, path, store, start_index=0, end_index=47, limit=None):
        path = Path(path)
        paths = sorted(path.iterdir()) if path.is_dir() else [path]
        frames = []
        for item in paths:
            if item.suffix == ".parquet":
                frames.append(pd.read_parquet(item))
            else:
                with item.open("rb") as stream:
                    frames.append(restricted_pickle_load(stream))
        self.rows = pd.concat(frames).reset_index()
        if limit is not None:
            self.rows = self.rows.iloc[:limit]
        self.store, self.start_index, self.end_index = store, start_index, end_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index >= len(self) or index < -len(self):
            raise IndexError(index)
        row = self.rows.iloc[index]
        ticks = self.store.day("backtest", row.instrument, row.date).index
        return Order(stock_id=row.instrument, amount=float(row.amount), direction=OrderDir(int(row.order_type)),
                     start_time=ticks[self.start_index], end_time=ticks[self.end_index - 1] + pd.Timedelta(minutes=1))
