# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Module contracts; intentionally separate from long learning experiments."""
import numpy as np
import pandas as pd
import pytest
import torch
from tianshou.data import Batch

from qlib.backtest.decision import Order, OrderDir
from qlib.rl.order_execution.regime import BayesianRegime, causal_features
from qlib.rl.order_execution.residual_data import FrameSimulator, ParquetStore
from qlib.rl.order_execution.residual_policy import ResidualPPO, masked_actor_loss
from qlib.rl.order_execution.residual_simulator import (
    ResidualExecutionSimulator, ResidualStateInterpreter, ResidualActionInterpreter,
    PotentialShapingReward, residual_amount,
)
from qlib.rl.order_execution.reward import PPOReward


def market_frame():
    ticks = pd.date_range("2021-01-04 09:30", periods=24, freq="5min").append(
        pd.date_range("2021-01-04 13:00", periods=24, freq="5min"))
    x = np.arange(48)
    return pd.DataFrame({"$close0": 100 + 0.03 * x + np.sin(x / 3),
                         "$volume0": 1000 + 10 * x}, index=ticks)


@pytest.fixture(scope="module")
def regime():
    rng = np.random.default_rng(42)
    sequences = [rng.normal(size=(n, 4)) for n in (48, 37, 28, 48, 19)]
    return BayesianRegime().fit(sequences, max_iter=8)


def high_budget(state):
    assert state.cur_step == len(state.history_steps)
    assert state.ticks_per_step == 6
    if state.cur_step == 7:
        return state.position
    return min(state.position, state.order.amount * (0.25 if state.cur_step % 2 else 0.0))


def order_for(frame, start=0, direction=OrderDir.SELL):
    return Order("SYNTH", 100.0, direction, frame.index[start], frame.index[46] + pd.Timedelta(minutes=1))


def test_market_cache_reuses_files_and_cannot_be_mutated(tmp_path, monkeypatch):
    import pickle

    frame = market_frame()
    reads = []

    def read(path):
        reads.append(path)
        return pd.concat({path.stem: frame}, names=["instrument", "datetime"])

    monkeypatch.setattr(pd, "read_parquet", read)
    store = ParquetStore(tmp_path, file_cache_size=256, day_cache_size=0)
    for _ in range(2):
        for stock in (f"STOCK{i}" for i in range(40)):
            pd.testing.assert_frame_equal(store.day("backtest", stock, frame.index[0]),
                                          frame.rename_axis("datetime"))
    assert len(reads) == 40  # revisiting >32 stocks must not decode every file again
    store.day_cache_size = 8
    day = store.day("backtest", "STOCK0", frame.index[0])
    day.iloc[0, 0] = -999
    assert store.day("backtest", "STOCK0", frame.index[0]).iloc[0, 0] == frame.iloc[0, 0]
    spawned = pickle.loads(pickle.dumps(store))
    assert not spawned._frames and not spawned._days and not spawned._regimes
    assert spawned.file_cache_size == 256
    uncached = ParquetStore(tmp_path, 0, 0, 0)
    pd.testing.assert_frame_equal(uncached.day("backtest", "STOCK0", frame.index[0]),
                                  store.day("backtest", "STOCK0", frame.index[0]))


@pytest.mark.parametrize("enabled", [False, True])
def test_cached_features_preserve_observations_rewards_and_metrics(regime, tmp_path, monkeypatch, enabled):
    frame = market_frame()
    monkeypatch.setattr(pd, "read_parquet",
                        lambda _: pd.concat({"SYNTH": frame}, names=["instrument", "datetime"]))
    store = ParquetStore(tmp_path)
    features = store.regime_features("SYNTH", frame.index[0], regime, enabled)
    assert store.regime_features("SYNTH", frame.index[0], regime, enabled) is features
    assert not features[0].flags.writeable and not features[1].flags.writeable
    assert store.regime_features("SYNTH", frame.index[0], regime, not enabled) is not features
    kwargs = dict(high_policy=high_budget, regime=regime, regime_enabled=enabled)
    cached = ResidualExecutionSimulator(order_for(frame), frame, market_features=features, **kwargs)
    original = ResidualExecutionSimulator(order_for(frame), frame, **kwargs)
    reward = PotentialShapingReward()
    tick = 0
    while not original.done():
        assert cached.done() == original.done()
        for key in original.get_state().observation:
            np.testing.assert_array_equal(cached.get_state().observation[key], original.get_state().observation[key])
        for sim in (original, cached):
            sim.step(tick % 5)
        assert reward(cached.get_state()) == reward(original.get_state())
        tick += 1
    assert cached.done()
    for key in ("pa", "ffr", "trade_price", "trade_value"):
        assert cached.inner.metrics[key] == original.inner.metrics[key]


@pytest.mark.parametrize("direction", [OrderDir.SELL, OrderDir.BUY])
@pytest.mark.parametrize("start", [0, 2])
@pytest.mark.parametrize("enabled,scale", [(False, 1.0), (True, 0.0)])
def test_disabled_and_zero_scale_reproduce_original(regime, direction, start, enabled, scale):
    frame = market_frame()
    order = order_for(frame, start, direction)
    original = FrameSimulator(order, frame)
    original_reward = PPOReward(8)
    original_return = 0.0
    while not original.done():
        original.step(high_budget(original.get_state()))
        original_return += original_reward.reward(original.get_state())
    changed = ResidualExecutionSimulator(order, frame, high_budget, regime,
                                         residual_enabled=enabled, residual_scale=scale)
    shaped_return = 0.0
    reward = PotentialShapingReward()
    while not changed.done():
        changed.step(4)  # largest residual must still be bypassed
        shaped_return += reward.reward(changed.get_state())
    np.testing.assert_allclose(original.history_exec.deal_amount, changed.inner.history_exec.deal_amount, atol=1e-10)
    np.testing.assert_allclose(original.history_steps.position, changed.macro_history.position, atol=1e-10)
    assert shaped_return == pytest.approx(original_return)
    for key in ("pa", "ffr", "trade_price", "trade_value"):
        assert original.metrics[key] == pytest.approx(changed.inner.metrics[key])


@pytest.mark.parametrize("direction,favorable_sign", [(OrderDir.SELL, 1), (OrderDir.BUY, -1)])
def test_arrival_reward_prices_each_fill_in_parent_order_bps(regime, direction, favorable_sign):
    frame = market_frame().iloc[:8].copy()
    frame["$close0"] = [85.0, 90.0, 100.0, 100.01, 99.99, 100.0, 100.02, 99.98]
    order = Order("SYNTH", 120.0, direction, frame.index[2], frame.index[-1] + pd.Timedelta(minutes=1))
    # Two macro intervals; the last interval must take all remaining inventory.
    sim = ResidualExecutionSimulator(order, frame, lambda _: 60.0, regime, base_reward="arrival_bps")
    actions = [0, 2, 2, 2, 0, 2]
    fills = [0, 30, 15, 15, 0, 60]
    expected = np.array([0.0, 0.25, -0.125, 0.0, 0.0, -1.0]) * favorable_sign
    unshaped, shaped = [], []
    reward = PotentialShapingReward()
    for action in actions:
        sim.step(action)
        state = sim.get_state()
        unshaped.append(state.raw_reward)
        shaped.append(reward.reward(state))
    assert sim.done()
    np.testing.assert_allclose(sim.inner.history_exec.deal_amount, fills, atol=1e-10)
    np.testing.assert_allclose(unshaped, expected, rtol=0, atol=1e-10)
    assert not np.allclose(shaped, unshaped)
    assert sum(shaped) == pytest.approx(sum(expected), abs=1e-10)
    assert sum(shaped) == pytest.approx(favorable_sign * (sim.inner.metrics["trade_price"] / 100 - 1) * 10_000)


@pytest.mark.parametrize("direction,sign", [(OrderDir.SELL, 1), (OrderDir.BUY, -1)])
def test_inventory_mark_prices_waiting_and_preserves_realized_arrival(regime, direction, sign):
    frame = market_frame().iloc[:9].copy()
    frame["$close0"] = [90., 95., 100., 100.01, 100.02, 100.04, 100.03, 100.07, 100.06]
    order = Order("SYNTH", 120., direction, frame.index[2], frame.index[-1] + pd.Timedelta(minutes=1))
    sim = ResidualExecutionSimulator(order, frame, lambda s: 0. if s.cur_step == 0 else s.position,
                                     regime, base_reward="arrival_bps", shaping_mode="inventory_mtm",
                                     shaping_coefficient=.5)
    shaped = PotentialShapingReward(True)
    raw_path, shaped_path, previous_phi = [], [], 0.
    # Four empty bars cross a window boundary; the first action in the last
    # window brings some of the later quantity forward.
    for act in [2, 2, 2, 2, 4, 2, 2]:
        sim.step(act)
        state = sim.get_state()
        assert state.phi_before == pytest.approx(previous_phi)
        previous_phi = state.phi_after
        raw_path.append(state.raw_reward)
        shaped_path.append(shaped.reward(state))
        if len(shaped_path) == 4:
            assert state.macro_boundary and not sim.done()
            assert previous_phi == pytest.approx(2. * sign)
    assert sim.done() and previous_phi == 0
    np.testing.assert_allclose(raw_path[:4], [0.] * 4, atol=1e-9)
    np.testing.assert_allclose(shaped_path[:4], sign * np.array([0., .5, .5, 1.]), atol=1e-9)
    expected = sign * 10000 * (sim.inner.metrics["trade_price"] / 100 - 1)
    assert sum(shaped_path) == pytest.approx(expected, abs=1e-8)
    assert sum(raw_path) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("direction,sign", [(OrderDir.SELL, 1), (OrderDir.BUY, -1)])
def test_inventory_shaping_keeps_fill_price_sensitivity_and_terminal_mark_zero(regime, direction, sign):
    frame = market_frame().iloc[:6].copy()
    frame["$close0"] = [100., 100.001, 100.002, 100.003, 100.004, 100.005]
    order = Order("SYNTH", 120., direction, frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1))
    def run(action):
        sim = ResidualExecutionSimulator(order, frame, lambda s: s.position, regime, residual_scale=4.,
                                         base_reward="arrival_bps", shaping_mode="inventory_mtm", shaping_coefficient=.5)
        sim.step(0)
        sim.step(action)
        state = sim.get_state()
        return sim, PotentialShapingReward(True).reward(state)
    early, take = run(4)
    waiting, hold = run(0)
    assert early.done() and early.get_state().phi_after == 0
    assert not waiting.done() and waiting.inner.position == pytest.approx(order.amount)
    # Same close, same start state: executing all rather than none retains
    # half the fill-price incentive, including moves well under one bp.
    assert take - hold == pytest.approx(sign * .05, abs=1e-9)


@pytest.mark.parametrize("direction,sign", [(OrderDir.SELL, 1), (OrderDir.BUY, -1)])
def test_window_pa_matches_separate_uniform_execution_after_early_finish(regime, direction, sign):
    frame = market_frame().iloc[:9].copy()
    frame["$close0"] = [70., 90., 100., 103., 98., 99., 107., 115., 123.]
    order = Order("SYNTH", 120., direction, frame.index[2], frame.index[-1] + pd.Timedelta(minutes=1))
    def budget(state):
        return min(60., state.position)
    def run(action):
        sim = ResidualExecutionSimulator(order, frame, budget, regime, residual_scale=9.,
            base_reward="window_pa", shaping_mode="relative_inventory_mtm", shaping_coefficient=.5)
        raw, shaped, marks = [], [], []
        while not sim.done():
            sim.step(action)
            state = sim.get_state()
            raw.append(state.raw_reward)
            shaped.append(PotentialShapingReward().reward(state))
            marks.append(state.phi_after)
            if state.macro_boundary:
                assert state.phi_after == 0
        return sim, np.array(raw), np.array(shaped), np.array(marks)
    neutral, neutral_raw, neutral_shaped, neutral_marks = run(2)
    np.testing.assert_allclose(neutral_marks, 0., atol=1e-10)
    np.testing.assert_allclose(neutral_raw, 0., atol=1e-9)
    np.testing.assert_allclose(neutral_shaped, 0., atol=1e-9)
    early, raw, shaped, marks = run(4)
    assert early.inner.history_exec.index[-1] == frame.index[6]  # does not visit prices 115 or 123
    expected = early.inner.metrics["pa"] - neutral.inner.metrics["pa"]
    independent = sign * 5000 * (107. - 115.) / frame["$close0"].iloc[2:].mean()
    assert expected == pytest.approx(independent)
    assert np.max(np.abs(marks)) > 0.
    assert sum(raw) == pytest.approx(expected, abs=1e-8)
    assert sum(shaped) == pytest.approx(expected, abs=1e-8)
    assert sum(w["delta_pa"] for w in early.window_stats.values()) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("direction,sign", [(OrderDir.SELL, 1), (OrderDir.BUY, -1)])
def test_window_pa_only_rewards_controlled_inventory_with_causal_inputs(regime, direction, sign):
    frame = market_frame().iloc[:8].copy()
    frame["$close0"] = [100., 105., 103., 102., 101., 98., 100., 100.01]
    other = frame.copy()
    other.iloc[-1, 0] += .02
    sims = []
    for f in (frame, other):
        order = Order("SYNTH", 120., direction, f.index[0], f.index[-1] + pd.Timedelta(minutes=1))
        sims.append(ResidualExecutionSimulator(order, f, lambda s: 0. if s.cur_step == 0 else s.position,
            regime, base_reward="window_pa", shaping_mode="relative_inventory_mtm", shaping_coefficient=.5))
    for _ in range(6):  # Zero high-level budget through a moving market.
        for sim in sims:
            sim.step(2)
            assert PotentialShapingReward().reward(sim.get_state()) == 0.
    for key, value in sims[0].get_state().observation.items():
        np.testing.assert_array_equal(value, sims[1].get_state().observation[key])
    for sim in sims:
        sim.step(4)  # Fill the two-bar window early.
        assert sim.done()
        assert sim.get_state().phi_after == 0.
    assert sign * sims[1].get_state().raw_reward < sign * sims[0].get_state().raw_reward < 0.


def test_shaping_telescopes_and_macro_boundaries_are_not_terminal(regime):
    frame = market_frame()
    sim = ResidualExecutionSimulator(order_for(frame), frame, high_budget, regime)
    enabled, disabled = PotentialShapingReward(), PotentialShapingReward(False)
    raw, shaped, boundaries, nonzero = 0.0, 0.0, 0, False
    rng = np.random.default_rng(5)
    while not sim.done():
        sim.step(int(rng.integers(5)))
        state = sim.get_state()
        raw += disabled.reward(state)
        shaped += enabled.reward(state)
        nonzero |= abs(state.phi_after - state.phi_before) > 1e-10
        if state.macro_boundary and not sim.done():
            boundaries += 1
            assert state.phi_after == 0
    assert boundaries >= 1
    assert nonzero
    assert shaped == pytest.approx(raw, abs=1e-10)
    assert sim.inner.metrics["ffr"] == pytest.approx(1.0)


def test_features_and_filter_cannot_see_future(regime):
    original = market_frame()
    altered = original.copy()
    altered.iloc[12:, 0] *= 3
    altered.iloc[12:, 1] *= 2
    a, b = causal_features(original), causal_features(altered)
    np.testing.assert_array_equal(a[:12], b[:12])
    np.testing.assert_allclose(regime.filter(a)[:12], regime.filter(b)[:12])
    sims = [ResidualExecutionSimulator(order_for(frame), frame, high_budget, regime) for frame in (original, altered)]
    for _ in range(12):
        for sim in sims:
            sim.step(2)
    for key in sims[0].get_state().observation:
        np.testing.assert_array_equal(sims[0].get_state().observation[key], sims[1].get_state().observation[key])


def test_regime_disable_keeps_same_market_normalization(regime):
    frame = market_frame()
    on = ResidualExecutionSimulator(order_for(frame), frame, high_budget, regime)
    off = ResidualExecutionSimulator(order_for(frame), frame, high_budget, regime, regime_enabled=False)
    for _ in range(10):
        on.step(2)
        off.step(2)
    assert not off.get_state().observation["regime"].any()
    np.testing.assert_array_equal(on.get_state().observation["market"], off.get_state().observation["market"])


def test_variational_objective_filter_and_serialization(regime, tmp_path):
    assert np.isfinite(regime.losses).all()
    assert np.all(np.diff(regime.losses) <= 1e-6)
    path = tmp_path / "regime.npz"
    regime.save(path)
    loaded = BayesianRegime.load(path)
    x = causal_features(market_frame())
    beliefs = loaded.filter(x)
    np.testing.assert_allclose(beliefs.sum(1), 1)
    np.testing.assert_allclose(beliefs, regime.filter(x))
    assert (regime.alpha >= regime.prior).all()


def test_residual_budget_and_last_tick():
    assert residual_amount(100, 100, 0, 5, 0, 1) == 20
    assert residual_amount(100, 60, 1, 5, 0, 1) == 0  # prior step ran ahead
    assert residual_amount(100, 100, 1, 5, 0, 1) == 40  # catch up
    assert residual_amount(100, 37, 4, 5, -1, 10) == 37
    assert residual_amount(0, 0, 0, 5, 1, 1) == 0


def test_forced_steps_have_no_actor_gradient():
    logits = torch.zeros(4, 5, requires_grad=True)
    dist = torch.distributions.Categorical(logits=logits)
    act = torch.tensor([0, 1, 2, 3])
    loss, entropy = masked_actor_loss(dist, act, dist.log_prob(act).detach(), torch.tensor([1., 2., 3., 4.]),
                                      torch.tensor([1, 0, 1, 0]), .2)
    (loss - .01 * entropy).backward()
    assert torch.count_nonzero(logits.grad[[1, 3]]) == 0
    assert torch.count_nonzero(logits.grad[[0, 2]]) > 0


def test_critic_still_learns_on_all_forced_steps():
    state, action = ResidualStateInterpreter(), ResidualActionInterpreter()
    policy = ResidualPPO(state.observation_space, action.action_space)
    obs = Batch(market=np.zeros((4, 12, 5), dtype=np.float32), regime=np.zeros((4, 4), dtype=np.float32),
                private=np.zeros((4, 8), dtype=np.float32), actor_valid=np.zeros((4, 1), dtype=np.float32))
    with torch.no_grad():
        dist = policy(Batch(obs=obs, info=Batch())).dist
        value = policy.critic(obs).clone()
        act = torch.full((4,), 2)
        old = dist.log_prob(act)
    batch = Batch(obs=obs, act=act, logp_old=old, adv=torch.arange(1., 5.), returns=value + 2, v_s=value)
    actor_before = policy.actor.logits.weight.detach().clone()
    critic_before = policy.critic.value_out.bias.detach().clone()
    policy.learn(batch, batch_size=4, repeat=1)
    torch.testing.assert_close(actor_before, policy.actor.logits.weight)
    assert not torch.equal(critic_before, policy.critic.value_out.bias)


def test_discount_cannot_silently_change_objective():
    with pytest.raises(ValueError, match="gamma=1"):
        PotentialShapingReward(discount_factor=0.99)
