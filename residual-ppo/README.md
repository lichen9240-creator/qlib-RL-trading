# Qlib 分层残差 PPO 下单实验

三次实验在同一批 640 笔开发测试订单上比较拆单策略，覆盖 20 只股票。原版高层 PPO 每 30 分钟分配执行量，低层残差 PPO 每 5 分钟调整拆单量。三个版本的完整模型都使用三状态 HMM，输入包括状态概率和熵。各版复用同一份高层 PPO 权重，低层训练 40 轮，每轮 128 笔订单，按验证集 PA 选择权重。

| 版本或模型 | 主要改动 | PA（bp）↑ | 成交率 | 到达价成本（bp）↓ | VWAP 滑点（bp）↓ |
|---|---|---:|---:|---:|---:|
| 截图版 · baseline | 微软原版 PPO | 4.241890 | 100% | −9.967298 | 0.596725 |
| 截图版 · residual_only | 加入低层残差 PPO，使用到达价奖励 | 4.358510 | 100% | −9.867212 | 0.492735 |
| 截图版 · residual_regime | 再加入 HMM 状态特征 | 4.557667 | 100% | −10.244798 | 0.282105 |
| 截图版 · full | 按剩余数量和剩余时间的偏差塑形，系数 0.1 | 4.099735 | 100% | −9.616613 | 0.748528 |
| 价格塑形 v1 · full | 保留到达价基础奖励，按剩余仓位的价格变化塑形，系数 0.5 | 4.362560 | 100% | −9.845002 | 0.490098 |
| 配对 PA v2 · full | 用同一高层预算下的均匀拆单作为窗口参照，按相对剩余仓位塑形，系数 0.5 | **5.257347** | **100%** | **−11.106330** | **−0.418871** |

表中 PA 是整单价格优势。v2 训练奖励使用的配对 PA 是实际拆单与均匀参照的 PA 之差。三套代码均含多进程采样、Parquet 缓存、高层批量推理和低层 GRU 整段计算。服务器配置为 16 个采样进程，高层推理每批最多 8 个请求。

## 源码、权重和原始结果

| 版本 | 版本专属源码 | 实验目录 |
|---|---|---|
| 截图中的四组实验 | [arrival_bps](versions/arrival_bps/) | [四组结果、配置与权重](results/iteration_progressive_20261114_205706_552014/) |
| 后续价格塑形 | [price_shaping_v1](versions/price_shaping_v1/) | [v1 结果、配置与权重](results/iteration_full_price_shaping_v1/) |
| 后续配对 PA | [paired_pa_v2](versions/paired_pa_v2/) | [v2 结果、配置与权重](results/iteration_full_paired_pa_v2/) |

每一版的实现由 `common/` 和对应的 `versions/<版本>/` 共同组成。后者保存该版的训练入口、执行模拟器、实验调度脚本和残差模块测试。两个目录中的相对路径和服务器工作区一致。微软 Qlib 的基础提交为 [54355232463878d2eebb91fe0ee5fa7fa1f5976c](https://github.com/microsoft/qlib/tree/54355232463878d2eebb91fe0ee5fa7fa1f5976c)。

每个实验目录的 `summary.csv` 是报告汇总，`order_execution_metrics.csv` 是逐单指标，`suite_settings.yml` 和各模型的 `resolved_config.yml` 保存实际参数。`best_validation.pth` 是相应模型在验证集上选出的权重。`sources_initial.json` 和 `stages/` 保存服务器记录的 SHA-256。另附的 `iteration_progressive_20261114_192841_287401` 目录提供了截图实验复用的基线输入。

## 在本地核对结果

在协作仓库根目录运行下面的命令。脚本只使用 Python 标准库，核对每版 14 个源文件及配置的哈希、训练阶段产物和订单集合，并从逐单记录重新计算表中的四项指标。

```bash
python3 residual-ppo/verify.py
```

## 在原实验环境中运行

在新的 Linux 工作区中准备微软原版 Qlib 和分层策略工作树，再复制公共文件及选定版本。以下示例选择配对 PA v2，使用原服务器路径，以便沿用归档配置中的绝对路径。从协作仓库根目录执行。

```bash
(
  set -eu
  package="$PWD/residual-ppo"
  workspace=/root/autodl-tmp/qlib-work
  version=paired_pa_v2
  base=54355232463878d2eebb91fe0ee5fa7fa1f5976c
  mkdir "$workspace"
  git clone https://github.com/microsoft/qlib.git "$workspace/qlib"
  git -C "$workspace/qlib" checkout --detach "$base"
  git -C "$workspace/qlib" worktree add --detach "$workspace/qlib-modules" "$base"
  cp -R "$package/common/." "$workspace/"
  cp -R "$package/versions/$version/." "$workspace/"
  cp -R "$package/results" "$workspace/"
)
```

训练和回测使用原项目已准备好的 5 分钟行情、特征及订单。将原数据目录放在 `qlib/examples/rl_order_execution/data/`，将 20 只股票的开发集订单放在 `datasets/iteration20_seed42/`。订单划分和校验值见各实验的 `orders_manifest.json`。原环境使用 `envs/qlib/`，软件版本见 [runtime-versions.txt](runtime-versions.txt)。

要重跑训练，在工作区根目录执行该版对应的命令。每个命令均以表中的前序实验作为输入。

| 已复制的版本 | 原训练命令 |
|---|---|
| `arrival_bps` | `bash run_full_experiment.sh --progressive-from results/iteration_progressive_20261114_192841_287401` |
| `price_shaping_v1` | `bash run_full_experiment.sh --full-only-from results/iteration_progressive_20261114_205706_552014` |
| `paired_pa_v2` | `bash run_full_experiment.sh --full-only-from results/iteration_full_price_shaping_v1` |

在已配好数据和环境的工作区中，可用保存的 v2 权重重新评估，结果写入独立目录。

```bash
workspace=/root/autodl-tmp/qlib-work
source "$workspace/activate.sh"
cd "$workspace/qlib-modules"
run="$workspace/results/iteration_full_paired_pa_v2"
PYTHONPATH="$PWD" "$workspace/envs/qlib/bin/python" -m qlib.rl.contrib.train_residual evaluate \
  --config "$run/full/resolved_config.yml" \
  --checkpoint "$run/full/best_validation.pth" \
  --split test --set "output=$workspace/results/paired_pa_v2_reevaluation"
```
