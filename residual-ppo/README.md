# Qlib 分层残差 PPO 的三个实验版本

高层 PPO 每 30 分钟分配执行量，低层 PPO 每 5 分钟调整拆单节奏。低层可以接收三状态隐马尔可夫模型的市场状态概率和熵。三个版本共享并行采样、行情缓存、高层批量推理和低层 GRU 整段计算的实现。

| 版本 | 训练方案 |
|---|---|
| [arrival_bps](versions/arrival_bps/) | 四组对照实验。从原版 PPO 开始，依次加入低层残差 PPO、市场状态输入和进度奖励塑形。低层的基础奖励按成交价格相对到达价的变化计算。 |
| [price_shaping_v1](versions/price_shaping_v1/) | 后续价格塑形版本。保留到达价基础奖励，用剩余仓位按市价估值的变化为低层提供逐步反馈。 |
| [paired_pa_v2](versions/paired_pa_v2/) | 后续配对 PA 版本。对每个高层窗口，以相同预算下的均匀拆单为参照，按实际执行相对参照的价格优势训练低层策略。 |

## 代码目录

每个版本都使用 [common](common/) 中的公共源码。版本目录保存对应的实验调度器 `scripts/full_experiment.py`、低层 PPO 训练入口 `qlib-modules/qlib/rl/contrib/train_residual.py`、执行模拟器和残差模块测试。三个版本的训练入口、模拟器和调度器都已与各次实验记录的 SHA-256 逐一核对。

## 使用配对 PA 版本

从协作仓库根目录执行以下命令，准备微软原版 Qlib 和对应的分层策略工作树。其他两版可以替换 `version`。

```bash
(
  set -eu
  package="$PWD/residual-ppo"
  workspace="$PWD/qlib-work"
  version=paired_pa_v2
  base=54355232463878d2eebb91fe0ee5fa7fa1f5976c
  mkdir "$workspace"
  git clone https://github.com/microsoft/qlib.git "$workspace/qlib"
  git -C "$workspace/qlib" checkout --detach "$base"
  git -C "$workspace/qlib" worktree add --detach "$workspace/qlib-modules" "$base"
  cp -R "$package/common/." "$workspace/"
  cp -R "$package/versions/$version/." "$workspace/"
)
```

训练或回测需要准备 Qlib 环境、5 分钟行情、订单和相应的实验配置。在 `qlib-modules/` 工作树中，低层训练和评估使用 `python -m qlib.rl.contrib.train_residual`，由 `--config` 指定配置。残差算法测试入口是 `python -m pytest tests/rl/test_residual_modules.py`。底层框架使用同一个 [微软 Qlib 提交](https://github.com/microsoft/qlib/tree/54355232463878d2eebb91fe0ee5fa7fa1f5976c)。
