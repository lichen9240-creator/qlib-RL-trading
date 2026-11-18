# qlib-RL-trading

Qlib + reinforcement learning quantitative trading: research, training and backtesting

本目录包含 Qlib 0.9.7 官方订单执行示例所用的数据、可运行代码、配置、模型和训练前后对比结果。原实验使用 RTX 5060 Laptop GPU、seed 42，运行 212 轮后触发官方提前停止。

训练效果、费用拆解、统计不确定性及下一轮优化方案见 [本次训练分析](ANALYSIS.md)。对比结果使用同一批 540 笔测试订单。

| 位置 | 内容 |
|---|---|
| `data/bin` | 原预处理选中的 10 只股票五分钟原始数据、日历及股票清单 |
| `data/pickle` | 原实验已生成的特征及模拟器数据 |
| `data/orders` | 固定训练、验证及测试订单 |
| `data/stat` | 原特征标准化统计量 |
| `data/manifest.json` | 187 个数据文件的大小和 SHA-256 校验值 |
| `code/train.py` | 调用官方入口，在 GPU 上开始一次新的 OPDS 训练 |
| `code/evaluate.py` | 运行训练前、第 12 轮、第 212 轮及 TWAP 的统一回测 |
| `code/verify_data.py` | 校验归档数据 |
| `code/analyze_results.py` | 重算训练效果、费用拆解、统计区间及分析报告 |
| `code/apply_compatibility.py` | 原环境使用的 Pandas、PyTorch 兼容修正 |
| `configs` | 相对路径训练配置、官方回测配置和原预处理配置快照 |
| `models/000.pth` | 按原 seed 42 和官方构造顺序重建的初始模型 |
| `models/012.pth` | 依据验证集选出的第 12 轮模型 |
| `models/212.pth` | 提前停止时的最终模型 |
| `models/checkpoints` | 原 OPDS 训练第 1 至第 212 轮的全部检查点 |
| `results/comparison.json` | 对比汇总、模型哈希、来源及初始化重建说明 |
| `results/comparison.csv` | 可直接读取的训练前后对比表 |
| `results/paired_orders.csv` | 四组模型逐订单匹配的价格优势、费用及成交比例 |
| `results/initialized`、`validation_best`、`final`、`twap` | 每组回测配置和完整结果 |
| `outputs/opds_comparison/comparison.xlsx` | Excel 对比表和逐订单明细 |
| `provenance` | 原训练日志、耗时记录、完整依赖锁定记录和数据处理源码 |
| `provenance/qlib_source` | 本次安装环境实际使用的 Qlib 强化学习源码快照及 MIT 许可 |

训练数据对应 2020-01-01 至 2021-06-30，验证数据对应 2021-07-01 至 2021-09-30，测试订单实际覆盖 2021-10-08 至 2021-12-30。原预处理按股票代码排序选取前 10 只股票，其中 SH600009 因最低成交量为零被官方订单生成器剔除，最终参与训练及测试的股票为 9 只。训练、验证、测试订单分别为 3,249、576、540 笔。每轮训练采集 10,000 个 episode，共采集 2,120,000 个训练 episode，另有 53 次验证。

训练前对比使用 seed 42 重建的初始化。原运行从第 1 轮起保存检查点，初始权重依据相同环境和官方初始化顺序重建。重建两次得到完全相同的 104 个状态张量，模型文件和重建方法记录在 `results/comparison.json`。第 12 轮依据验证集 PA 选择，训练结束时使用第 212 轮权重。四组回测使用相同的 540 笔订单、数据、费用及整手约束。

表中的 PA 衡量执行价格相对订单时间窗口 TWAP 的优势，数值越高越好，1 bp 等于 0.01%。扣费后 PA 按每笔实际交易费用折算为基点后计算。成交金额加权 PA 按成交数量乘基准价格加权。回测采用买入费率 0.05%、卖出费率 0.15%、最低费用 5 元、交易单位 100 股。本次订单全部为卖出方向。Excel 的变化列以训练前为基准，Excel 内的逐订单计算和变化列使用公式。

在 PowerShell 中进入本目录，完成环境安装后执行以下命令。已有 Python 3.12 环境可以通过 `run.ps1` 的 `-Python` 参数指定。

```powershell
.\run.ps1 -Stage verify
.\run.ps1 -Stage evaluate
.\run.ps1 -Stage analyze
```

评估会更新 `results` 内的四组回测及 JSON、CSV 对比结果。Excel 归档保存本次计算结果，生成脚本为 `code/build_comparison.mjs`，使用 Codex 捆绑的 Artifact Tool 运行时。

开始一次新的 GPU 训练时运行以下命令。新任务写入 `runs/train_<时间>`，会在采集第一条 episode 前保存 `initial_model.pth`。

```powershell
.\run.ps1 -Stage train
```

复制到另一台 Windows 机器后，可使用下面的命令创建本地 Python 3.12 环境并安装依赖。`requirements.txt` 固定主要训练依赖，原运行完整依赖列表保存在 `provenance/requirements-lock.txt`。安装脚本会应用兼容修正并检查 CUDA。

```powershell
.\setup.ps1
.\run.ps1 -Stage verify
.\run.ps1 -Stage evaluate
```

本次整理已验证原环境下的数据校验和四组回测。新环境安装脚本已完成语法检查，全新环境安装尚待实际验证。原始预处理脚本保存在 `provenance/original_preparation`，归档包含原实验已生成的固定订单。重新生成订单会使用新的随机抽样，逐文件复现应使用本目录提供的数据。

原训练的训练及验证墙钟耗时为 39 小时 51 分 27 秒，其中 Windows 记录的睡眠合计 12 小时 37 分 8 秒。扣除睡眠后约为 27 小时 14 分 19 秒。原自动测试约 34 秒。训练退出时出现队列清理阶段的 `handle is closed` 告警，检查点、540 条测试结果及独立回测均已核验。完整记录见 `provenance/original_backtest_summary.json`。

本次结论对应 9 只股票、单随机种子的订单执行实验。原数据来自 Qlib 示例所用的五分钟数据包，下载来源、原始压缩包哈希及整理后的数据哈希均保存在 `provenance/data_manifest.json` 和 `data/manifest.json`。
