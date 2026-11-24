from pathlib import Path
import os,pickle,runpy,sys,json
import yaml
import pandas as pd
import joblib
import qlib.contrib.data.highfreq_provider as provider
root=Path('/root/autodl-tmp/qlib-work')
example=root/'qlib/examples/rl_order_execution'
os.chdir(example)
conf=yaml.load((example/'scripts/pickle_data_config.yml').read_text(),Loader=yaml.FullLoader)
conf['qlib_conf']['kernels']=8
with (root/'configs/pickle_data.yml').open('w') as f: yaml.dump(conf,f)
original_parallel=joblib.Parallel
def bounded_parallel(*args,**kwargs):
    kwargs['n_jobs']=min(kwargs.get('n_jobs',8),8)
    return original_parallel(*args,**kwargs)
provider.Parallel=bounded_parallel
sys.argv=['gen_pickle_data.py','-c',str(root/'configs/pickle_data.yml')]
runpy.run_path(str(example/'scripts/gen_pickle_data.py'),run_name='__main__')
runpy.run_path(str(root/'scripts/gen_training_orders_compat.py'),run_name='__main__')
# Upstream merge_orders.py uses pickle.dump but omits its pickle import.
runpy.run_path(str(example/'scripts/merge_orders.py'),init_globals={'pickle':pickle},run_name='__main__')
orders=example/'data/orders'
preview=example/'data/orders_preview'
preview.mkdir(exist_ok=True)
if not (preview/'train').exists(): (preview/'train').symlink_to(orders/'train',target_is_directory=True)
counts={}
for split in ['train','valid','test']:
    frames=[pd.read_pickle(p) for p in sorted((orders/split).iterdir()) if p.is_file()]
    full=pd.concat(frames).sort_index()
    counts[split]={'full_orders':len(full),'stocks':len(full.index.get_level_values('instrument').unique())}
    if split!='train':
        subset=full.sample(n=min(1024,len(full)),random_state=42).sort_index()
        (preview/split).mkdir(exist_ok=True)
        subset.to_pickle(preview/split/'preview.pkl.target')
        counts[split]['preview_orders']=len(subset)
        if split=='test': subset.drop(columns=['$close0'],errors='ignore').to_pickle(preview/'test_orders.pkl')
(root/'results/data_manifest.json').write_text(json.dumps(counts,indent=2))
base=yaml.safe_load((example/'exp_configs/train_ppo.yml').read_text())
base['runtime']['use_cuda']=True
base['env']['concurrency']=8
base['data']['source']['order_dir']='./data/orders_preview'
base['trainer'].update(max_epoch=10,episode_per_collect=512,val_every_n_epoch=2,checkpoint_path=str(root/'results/ppo_preview'))
(root/'configs/train_ppo_preview.yml').write_text(yaml.safe_dump(base,sort_keys=False))
full=yaml.safe_load((example/'exp_configs/train_ppo.yml').read_text())
full['runtime']['use_cuda']=True
full['trainer']['checkpoint_path']=str(root/'results/ppo_full')
(root/'configs/train_ppo_full.yml').write_text(yaml.safe_dump(full,sort_keys=False))
print('DATA_PREPARATION_COMPLETE',json.dumps(counts),flush=True)
