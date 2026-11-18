"""Bounded run of the official Qlib v0.9.7 order execution example."""
import copy
import json
import os
from pathlib import Path
import time

import yaml
import qlib
from qlib.contrib.data.highfreq_provider import HighFreqProvider
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / 'example'


def main():
    os.chdir(EXAMPLE)
    conf = yaml.load(Path('scripts/pickle_data_config.yml').read_text(), Loader=yaml.FullLoader)
    # Deterministic universe selection; no outcome-based filtering.
    instrument_file = Path('data/bin/instruments/csi300s19_22.txt')
    stocks = sorted({line.split()[0] for line in instrument_file.read_text().splitlines()})[:10]
    conf['instruments'] = stocks
    conf['qlib_conf'].update(kernels=2, joblib_backend='threading')
    for name in ['feature_conf', 'backtest_conf']:
        conf[name]['kwargs']['handler']['kwargs']['instruments'] = stocks
    Path('run_configs').mkdir(exist_ok=True)
    Path('run_configs/pickle_data.yml').write_text(yaml.dump(conf), encoding='utf-8')
    provider = HighFreqProvider(**conf)
    started = time.perf_counter()
    # Same global normalization and stock splitting as the official generator,
    # performed serially instead of spawning 32 Windows processes.
    for name, kind in [('feature_conf', 'feature'), ('backtest_conf', 'backtest')]:
        config = copy.deepcopy(conf[name])
        provider._gen_dataframe(copy.deepcopy(config))
        destination = Path(config.pop('path')).with_suffix('')
        destination.mkdir(parents=True, exist_ok=True)
        dataset = init_instance_by_config(config)
        for stock in stocks:
            output = destination / f'{stock}.pkl'
            if output.exists():
                continue
            dataset.handler.config(instruments=[stock])
            if kind == 'backtest':
                dataset.handler.setup_data()
            else:
                dataset.handler.setup_data(init_type=DataHandlerLP.IT_LS)
            dataset.config(dump_all=True, recursive=True)
            dataset.to_pickle(output)
            print('PREPARED', kind, stock, flush=True)
    train = yaml.safe_load(Path('exp_configs/train_ppo.yml').read_text())
    original = copy.deepcopy(train)
    train['runtime']['use_cuda'] = True
    train['env']['concurrency'] = 8
    train['trainer'].update(max_epoch=20, repeat_per_collect=10,
                            episode_per_collect=512, batch_size=256,
                            val_every_n_epoch=2, earlystop_patience=20)
    Path('run_configs/train_ppo.yml').write_text(yaml.safe_dump(train), encoding='utf-8')
    (ROOT / 'run_plan.json').write_text(json.dumps({
        'source_version': qlib.__version__, 'stocks': stocks,
        'data_split': {'train': ['2020-01-01','2021-06-30'],
                       'validation': ['2021-07-01','2021-09-30'],
                       'test': ['2021-10-01','2021-12-31']},
        'original_training_config': original, 'run_training_config': train,
        'preparation_seconds': time.perf_counter()-started,
        'scope': 'Official PPO network/reward/action with a deterministic 10-stock subset and reduced budget; not a full benchmark reproduction'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
