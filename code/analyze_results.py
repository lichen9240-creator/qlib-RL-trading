"""Reproduce the OPDS learning-curve, fee and paired-order analysis."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_NAMES = {
    "initialized": "训练前初始化",
    "validation_best": "第 12 轮验证最佳",
    "final": "第 212 轮最终模型",
    "twap": "TWAP",
}
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 10000


def summarize_model(frame):
    notional = frame["deal_amount"].abs() * frame["base_price"]
    fee_bps = frame["trade_cost"] / notional * 10000
    proportional_fees = frame["trade_value"].abs() * 0.0015 / notional * 10000
    net_pa = frame["pa"] - fee_bps
    return {
        "orders": len(frame),
        "mean_pa_bps": float(frame["pa"].mean()),
        "mean_net_pa_bps": float(net_pa.mean()),
        "weighted_pa_bps": float(np.average(frame["pa"], weights=notional)),
        "weighted_net_pa_bps": float(np.average(net_pa, weights=notional)),
        "mean_fee_bps": float(fee_bps.mean()),
        "proportional_fee_bps": float(proportional_fees.mean()),
        "minimum_fee_excess_bps": float((fee_bps - proportional_fees).mean()),
        "minimum_fee_affected_orders": int(
            (frame["trade_cost"] - frame["trade_value"].abs() * 0.0015 > 1e-6).sum()
        ),
        "pa_standard_deviation_bps": float(frame["pa"].std()),
        "mean_fulfillment_ratio": float(frame["ffr"].mean()),
    }


def paired_comparison(left, right):
    assert left.index.equals(right.index)
    differences = left["pa"] - right["pa"]
    daily = differences.groupby(level="datetime").mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = rng.choice(
        daily.to_numpy(), size=(BOOTSTRAP_SAMPLES, len(daily)), replace=True
    ).mean(axis=1)
    per_stock = differences.groupby(level="instrument").mean()
    return {
        "mean_gross_pa_difference_bps": float(differences.mean()),
        "order_win_fraction": float((differences > 0).mean()),
        "day_win_fraction": float((daily > 0).mean()),
        "trading_days": len(daily),
        "daily_cluster_bootstrap_95pct_ci_bps": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "stock_mean_differences_bps": per_stock.to_dict(),
        "stock_contributions_to_overall_mean_bps": (per_stock / len(per_stock)).to_dict(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def draw_curves(training, validation):
    try:
        import matplotlib
    except ModuleNotFoundError:
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.6, 5.0), layout="constrained")
    axis.plot(training["round"], training["pa"], color="#304F73", label="Training PA")
    axis.plot(
        validation["round"], validation["val/pa"], color="#B06326",
        marker="o", markersize=2.5, label="Validation PA",
    )
    axis.axhline(0, color="#9BA2AB", linewidth=0.8)
    axis.axvline(12, color="#667785", linewidth=0.8, linestyle="--")
    axis.scatter([12], [validation.loc[validation["round"] == 12, "val/pa"].iloc[0]],
                 color="#B06326", s=35, zorder=3)
    axis.set(xlabel="Training round", ylabel="Price advantage / bp",
             title="OPDS training and validation price advantage")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    target = RESULTS / "training_curves.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target.relative_to(ROOT).as_posix()


def write_report(analysis):
    models = analysis["models"]
    best = models["validation_best"]
    final = models["final"]
    twap = models["twap"]
    paired = analysis["paired_gross_pa"]["validation_best_minus_final"]
    low, high = paired["daily_cluster_bootstrap_95pct_ci_bps"]
    rows = []
    for key, label in MODEL_NAMES.items():
        item = models[key]
        rows.append(
            f"| {label} | {item['mean_pa_bps']:+.4f} | {item['mean_net_pa_bps']:+.4f} "
            f"| {item['weighted_pa_bps']:+.4f} | {item['weighted_net_pa_bps']:+.4f} |"
        )
    report = "\n".join([
        "# OPDS 训练效果分析",
        "",
        "本次训练在训练集的执行指标上改善，后期验证表现出现下降。第 12 轮在订单等权平均指标上更好，第 212 轮在成交金额加权指标上更好。选择模型时应先确定评价目标。以下数字来自同一批 9 只股票、540 笔测试订单，测试实际覆盖 2021-10-08 至 2021-12-30。数值单位为 bp，1 bp 等于 0.01%，价格优势越高越好。",
        "",
        "| 模型 | 等权 PA | 等权扣费后 PA | 金额加权 PA | 金额加权扣费后 PA |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        "训练 PA 从第 12 轮的 +0.3865 bp 上升到第 212 轮的 +11.7953 bp，验证 PA 同时从 +6.1887 bp 降到 +0.7154 bp。验证集排名前十的检查点中，九个位于前 36 轮。训练和验证曲线的分化与过拟合、时间分布变化相符。本次单随机种子实验无法把两者的影响单独识别。训练时采样动作、验证时确定性决策也会影响两条曲线的绝对差距。",
        "",
        "官方早停的 patience 为 50，按验证次数计数，每 4 轮验证一次。因此第 12 轮刷新最佳后，又经过 200 轮才在第 212 轮停止。默认 restore_best_weights 为 False，原自动测试使用第 212 轮权重。第 12 轮根据验证集选出，随后单独在固定测试集上回测。[官方早停实现](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/rl/trainer/callbacks.py)",
        "",
        f"第 12 轮比第 212 轮的订单等权 PA 高 {paired['mean_gross_pa_difference_bps']:.4f} bp。SH600004 的股票内平均差为 +17.8678 bp，对整体均值贡献 +1.9853 bp，其余八只股票合计贡献约 −0.2325 bp。按 60 个交易日聚类重抽样的 95% 区间为 [{low:.4f}, {high:.4f}] bp，跨过零。样本对稳定优势的支持有限。该区间覆盖同日股票之间的共同变化，跨日序列相关性、模型选择及不同训练随机种子的变动仍需另外评估。",
        "",
        f"第 12 轮的扣费后结果为 {best['mean_pa_bps']:+.4f} − {best['mean_fee_bps']:.4f} = {best['mean_net_pa_bps']:+.4f} bp。配置中的卖出费率为 0.15%，约贡献 {best['proportional_fee_bps']:.4f} bp，最低费用额外贡献 {best['minimum_fee_excess_bps']:.4f} bp。价格改善的量级小于费用，扣费后均值因此仍为负。此处费用来自本次示例配置，实际账户费用需要按真实交易规则重新设置。",
        "",
        f"TWAP 的等权平均费用为 {twap['mean_fee_bps']:.4f} bp，最低费用额外贡献 {twap['minimum_fee_excess_bps']:.4f} bp。第 12 轮相对 TWAP 的等权扣费后改善为 {best['mean_net_pa_bps']-twap['mean_net_pa_bps']:.4f} bp，其中约 {twap['mean_fee_bps']-best['mean_fee_bps']:.4f} bp 来自费用差异。小订单拆分产生的最低费用对等权结果影响很大。金额加权扣费后 PA 则为第 12 轮 −16.9071 bp、第 212 轮 −15.7389 bp、TWAP −15.9635 bp，实际金额权重改变了模型排序。",
        "",
        "OPDS 的 PAPenaltyReward 组合价格优势和成交量集中度惩罚，本次 scale 为 0.01、penalty 为 4。奖励中没有回测所用的最低费用及整手处理，训练还使用简化模拟器。官方文档说明训练和回测的执行约束存在差异。对每笔固定数量、最终全部成交的订单，单纯给所有策略减去相同固定费率，通常无法改变最优动作，成本优化应处理会随执行方式变化的费用及冲击。[官方奖励实现](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/rl/order_execution/reward.py) [官方示例说明](https://github.com/microsoft/qlib/blob/v0.9.7/examples/rl_order_execution/README.md)",
        "",
        "后期策略的验证平均步数由第 12 轮的 5.29 降为 3.67，最低费用影响的测试订单由 300 笔降为 20 笔，测试 PA 标准差从 42.15 增至 62.55 bp。这些现象显示执行行为和价格风险发生变化。确认具体交易时点、拆单次数及原因，需要在下一次运行保存逐步动作和成交日志。",
        "",
        "接下来可以按以下顺序开展小规模实验。",
        "",
        "1. 先明确按订单等权还是按成交金额加权优化，模型选择指标采用相应的同费用 TWAP 净节省。自动保存并加载该指标的验证最佳权重，尝试把早停 patience 调整为 5–10 次验证，并同步记录两种权重下的结果。",
        "2. 用更接近回测的训练执行约束处理最低费用、100 股整手、未完成订单和流动性，评估会随执行路径改变的成本。增加逐步动作及成交日志，检查当前 0%、25%、50%、75%、100% 整笔订单比例的动作空间是否过粗。",
        "3. 先对学习率 1e−4 与 3e−5、每轮更新重复次数 5 与 10 做 2×2 的短程筛选，再对候选设置使用 3–5 个随机种子和滚动时间验证。原训练每轮重复更新 25 次，可以检验减少重复拟合是否有利于样本外表现。",
        "4. 扩大 9 只股票和单一时间切分的数据范围，纳入不同流动性及市场阶段。当前测试结果已经被查看，下一轮调参使用新的验证窗口，最终评价另留未来时间窗口。",
        "",
        "本次扣除睡眠的训练及验证共约 27 小时 14 分，平均每轮约 7.71 分钟。按历史速度做保守预算，单个 24–32 轮试验约需 3.1–4.1 小时，2×2 筛选约需 12.3–16.4 小时。减少重复更新可能缩短耗时，环境模拟和机器状态也会影响速度，实际预算应由每项试验前几轮重新估计。多随机种子阶段再针对筛选出的少量设置运行。",
        "",
        "本报告依据本次冻结数据和本地回测计算，初始模型由原环境及 seed 42 重建。540 笔测试订单均已匹配，成交比例均约 99.71%。费用节省、价格风险和泛化表现需要结合目标订单规模共同判断。",
        "",
        "完成项目环境安装后，运行下面的命令可以重建 JSON 和本报告。环境已安装 matplotlib 时还会更新训练曲线图。",
        "",
        "```powershell",
        ".\\run.ps1 -Stage verify",
        ".\\run.ps1 -Stage analyze",
        "```",
        "",
        "重建脚本为 [code/analyze_results.py](code/analyze_results.py)，计算结果为 [results/training_analysis.json](results/training_analysis.json)。在其他机器上可以使用本项目环境的 Python 执行同一脚本。",
    ])
    if analysis["plot"]:
        report += "\n\n![训练与验证价格优势](results/training_curves.png)\n"
    (ROOT / "ANALYSIS.md").write_text(report + "\n", encoding="utf-8")


def main():
    training = pd.read_csv(RESULTS / "train_result.csv")
    validation = pd.read_csv(RESULTS / "validation_result.csv")
    training["round"] = np.arange(1, len(training) + 1)
    validation["round"] = np.arange(1, len(validation) + 1) * 4
    assert len(training) == 212 and len(validation) == 53
    source = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
    frames = {}
    for name in MODEL_NAMES:
        frame = pd.read_csv(
            RESULTS / name / "backtest_result.csv", index_col=[0, 1], parse_dates=[1]
        ).sort_index()
        assert len(frame) == 540 and frame.index.is_unique
        assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
        frames[name] = frame
    models = {name: summarize_model(frame) for name, frame in frames.items()}
    for name, record in models.items():
        assert np.isclose(record["mean_pa_bps"], source["models"][name]["mean_pa_bps"])
        assert np.isclose(record["mean_net_pa_bps"], source["models"][name]["mean_fee_adjusted_pa_bps"])
    timing = source["provenance"]["original_time_accounting"]
    seconds_per_round = timing["TrainingAndValidationExcludingSleepSeconds"] / len(training)
    inputs = [RESULTS / "train_result.csv", RESULTS / "validation_result.csv", RESULTS / "comparison.json"]
    inputs.extend(RESULTS / name / "backtest_result.csv" for name in MODEL_NAMES)
    analysis = {
        "models": models,
        "training_selected_rounds": training.loc[
            training["round"].isin([1, 12, 40, 80, 120, 160, 200, 212]),
            ["round", "pa", "reward", "steps_per_episode"],
        ].to_dict("records"),
        "validation_curve": validation[
            ["round", "val/pa", "val/reward", "val/steps_per_episode"]
        ].to_dict("records"),
        "paired_gross_pa": {
            "validation_best_minus_final": paired_comparison(frames["validation_best"], frames["final"]),
            "validation_best_minus_initialized": paired_comparison(frames["validation_best"], frames["initialized"]),
        },
        "pilot_budget": {
            "historical_awake_seconds_per_round": seconds_per_round,
            "single_24_round_hours": seconds_per_round * 24 / 3600,
            "single_32_round_hours": seconds_per_round * 32 / 3600,
            "four_24_round_hours": seconds_per_round * 24 * 4 / 3600,
            "four_32_round_hours": seconds_per_round * 32 * 4 / 3600,
            "basis": "Historical average with the original 25 update repeats; provisional until a new pilot is measured",
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "plot": draw_curves(training, validation),
        "scope": "One seed, nine stocks, 60 test days; bootstrap does not include all temporal or model-selection uncertainty",
    }
    (RESULTS / "training_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_report(analysis)
    print(json.dumps({"report": "ANALYSIS.md", "analysis": "results/training_analysis.json", "plot": analysis["plot"]}))


if __name__ == "__main__":
    main()
