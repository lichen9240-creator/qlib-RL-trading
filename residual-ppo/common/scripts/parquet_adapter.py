"""Data-only adapter for the previously converted, fixed Qlib example data."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import qlib.rl.data.native as native

ROOT = Path('/root/autodl-tmp/qlib-work/qlib/examples/rl_order_execution/data')


def _read_frame(path):
    return pd.read_parquet(path)


def _read_day(path, stock_id, start):
    frame = read_frame(path)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    data = frame.loc[pd.IndexSlice[stock_id, start:end], :].reset_index()
    if 'instrument' in data.columns:
        data = data.drop(columns=['instrument'])
    return data.set_index('datetime')


read_frame = lru_cache(maxsize=256)(_read_frame)
read_day = lru_cache(maxsize=4096)(_read_day)


class ParquetIntradayProcessedData(native.HandlerIntradayProcessedData):
    def __init__(self, data_dir, stock_id, date, feature_columns_today,
                 feature_columns_yesterday, backtest=False, index_only=False):
        if Path(data_dir).resolve() != (ROOT / 'pickle').resolve():
            raise ValueError(f'Unexpected data root: {data_dir}')
        path = (ROOT / 'parquet' / ('backtest' if backtest else 'feature') / f'{stock_id}.parquet').resolve()
        if not path.is_relative_to((ROOT / 'parquet').resolve()):
            raise ValueError(f'Invalid stock path: {stock_id}')
        data = read_day(str(path), stock_id, pd.Timestamp(date).normalize())
        # Preserve the original per-instance mutable today/yesterday frames.
        self.today = data[[] if index_only else feature_columns_today].copy()
        self.yesterday = data[[] if index_only else feature_columns_yesterday].copy()


def install(file_cache_size=256, day_cache_size=4096):
    """Install separately in the parent and each spawned worker; zero disables."""
    global read_frame, read_day
    if any(type(n) is not int or n < 0 for n in (file_cache_size, day_cache_size)):
        raise ValueError('Cache sizes must be nonnegative integers')
    read_frame = lru_cache(maxsize=file_cache_size)(_read_frame)
    read_day = lru_cache(maxsize=day_cache_size)(_read_day)
    native.HandlerIntradayProcessedData = ParquetIntradayProcessedData
