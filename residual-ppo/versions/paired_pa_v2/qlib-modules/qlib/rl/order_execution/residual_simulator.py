# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Frozen macro PPO + micro TWAP residuals on Qlib's Simple execution backend.

The supported baseline has no volume cap, lot rounding or market impact.
Those are properties of the existing Simple experiment, not new assumptions
silently added to the native exchange backtest.
"""
from dataclasses import dataclass

import numpy as np
import torch
from gym import spaces
from tianshou.data import Batch

from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import Simulator
from qlib.rl.order_execution.reward import PPOReward
from qlib.rl.order_execution.regime import causal_features
from qlib.rl.order_execution.residual_data import FrameSimulator
from qlib.rl.order_execution.state import SAOEState
from qlib.rl.order_execution.utils import price_advantage


BASE_REWARDS = ("ppo_discrete", "arrival_bps", "window_pa")
SHAPING_MODES = ("schedule", "inventory_mtm", "relative_inventory_mtm")


class FrozenHighPolicy:
    def __init__(self, policy, state_interpreter, action_interpreter):
        self.policy = policy.eval()
        self.policy.requires_grad_(False)
        self.state_interpreter, self.action_interpreter = state_interpreter, action_interpreter

    def __call__(self, state):
        obs = self.state_interpreter.interpret(state)
        action = self.predict_observation(obs)
        return float(self.action_interpreter.interpret(state, action))

    def predict_observation(self, obs):
        """Reference path, used when high_batch_size=1 or sampling is serial."""
        with torch.no_grad():
            action = self.policy(Batch(obs=Batch([obs]), info=Batch())).act[0]
        return int(action)

    def predict_observations(self, observations):
        """Batch already-ready requests; transfer the action vector to CPU once."""
        if not observations:
            return []
        with torch.no_grad():
            actions = self.policy(Batch(obs=Batch(observations), info=Batch())).act
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        return np.asarray(actions).reshape(-1).astype(np.int64).tolist()


def residual_amount(budget, remaining, index, total, delta, scale):
    """Cumulative TWAP reference with catch-up after an earlier residual action."""
    if remaining <= 1e-8:
        return 0.0
    if index == total - 1:
        return float(remaining)
    target = budget * (index + 1) / total - (budget - remaining)
    return float(np.clip(target + delta * scale * budget / total, 0, remaining))


def potential(budget, remaining, index, total, coefficient):
    if budget <= 1e-8 or index == 0 or index >= total:
        return 0.0
    return -coefficient * (remaining / budget - (total - index) / total) ** 2


@dataclass(frozen=True)
class ResidualState:
    execution: SAOEState
    observation: dict
    raw_reward: float = 0.0
    phi_before: float = 0.0
    phi_after: float = 0.0
    macro_boundary: bool = False


class ResidualExecutionSimulator(Simulator):
    def __init__(self, order, frame, high_policy, regime, regime_enabled=True, residual_enabled=True,
                 residual_scale=1.0, residual_values=(-1, -0.5, 0, 0.5, 1), shaping_coefficient=0.1,
                 window=12, macro_minutes=30, data_granularity=5, max_step=8, market_features=None,
                 base_reward="ppo_discrete", shaping_mode="schedule"):
        super().__init__(initial=order)
        if macro_minutes % data_granularity or residual_scale < 0 or shaping_coefficient < 0:
            raise ValueError("Invalid execution hyperparameters")
        if not (len(residual_values) >= 3 and residual_values.count(0) == 1):
            raise ValueError("Residual actions require exactly one zero action")
        if base_reward not in BASE_REWARDS:
            raise ValueError(f"Unknown base_reward {base_reward!r}; expected one of {BASE_REWARDS}")
        if shaping_mode not in SHAPING_MODES:
            raise ValueError(f"Unknown shaping mode: {shaping_mode}")
        if shaping_mode == "inventory_mtm" and (base_reward != "arrival_bps" or not 0 <= shaping_coefficient <= 1):
            raise ValueError("Inventory shaping requires arrival_bps and a weight between 0 and 1")
        if shaping_mode == "relative_inventory_mtm" and (base_reward != "window_pa" or not 0 <= shaping_coefficient <= 1):
            raise ValueError("Relative inventory shaping requires window_pa and a weight between 0 and 1")
        self.inner = FrameSimulator(order, frame, data_granularity, data_granularity)
        self.base_reward = base_reward
        self.shaping_mode = shaping_mode
        self.inventory_phi = 0.0  # Start the order at its arrival mark.
        if base_reward == "window_pa":
            if not (np.isfinite(order.amount) and order.amount > 0 and
                    np.isfinite(self.inner.twap_price) and self.inner.twap_price > 0):
                raise ValueError("Window PA requires a positive parent size and a positive, finite order TWAP")
            self.uniform_value = 0.0
            self.raw_total = 0.0
            self.window_stats = {}
        if base_reward == "arrival_bps":
            # Same arrival proxy as the IS report: the first executable bar of this order.
            self.arrival_price = float(self.inner.backtest_data.get_deal_price().at[self.inner.cur_time])
            if not np.isfinite(self.arrival_price) or self.arrival_price <= 0:
                raise ValueError(f"Invalid arrival price for {order.stock_id}: {self.arrival_price}")
            if not np.isfinite(order.amount) or order.amount <= 0:
                raise ValueError(f"Invalid parent order size for {order.stock_id}: {order.amount}")
        self.high_policy, self.regime = high_policy, regime
        self.regime_enabled = regime_enabled
        self.enabled, self.scale, self.values = residual_enabled, residual_scale, tuple(residual_values)
        self.coefficient, self.window = shaping_coefficient, window
        self.macro_ticks = macro_minutes // data_granularity
        self.macro_step = 0
        self.residual_decisions = 0
        self.nonzero_residual_actions = 0
        self.changed_residual_amounts = 0
        self.macro_history = self.inner.history_steps.copy()
        self.original_reward = PPOReward(max_step=max_step) if base_reward == "ppo_discrete" else None
        self.max_step = max_step
        # Vectorized features/filter are causal; observations slice strictly before cur_time.
        if market_features is None:
            raw = causal_features(frame)
            self.features = regime.normalize(raw).astype(np.float32)
            self.beliefs = regime.filter(raw) if regime_enabled else np.zeros((len(raw), regime.n_states))
        else:
            self.features, self.beliefs = market_features
            if self.features.shape != (len(frame), 4) or self.beliefs.shape != (len(frame), regime.n_states):
                raise ValueError("Cached market features do not match this stock-day")
        self.transition = (0.0, 0.0, 0.0, False)
        self._start_macro()

    def _macro_state(self):
        return self.inner.get_state()._replace(cur_step=self.macro_step, history_steps=self.macro_history,
                                               ticks_per_step=self.macro_ticks)

    def _start_macro(self):
        self.macro_start = self.inner.cur_time
        loc = self.inner.ticks_index.get_loc(self.macro_start)
        next_loc = (loc // self.macro_ticks + 1) * self.macro_ticks
        self.macro_end = (self.inner.ticks_index[next_loc] if next_loc < len(self.inner.ticks_index)
                          else self.inner.order.end_time)
        self.macro_end = min(self.macro_end, self.inner.order.end_time)
        bars = self.inner._get_ticks_slice(self.macro_start, self.macro_end)
        self.total = len(bars)
        self.budget = float(self.high_policy(self._macro_state()))
        if not np.isfinite(self.budget) or self.budget < 0 or self.budget > self.inner.position + 1e-6:
            raise ValueError("Frozen high-level policy returned an invalid budget")
        if self.macro_end >= self.inner.order.end_time:
            # This is the original Simple backend's final-fill rule, even for a zero high action.
            self.budget = self.inner.position
        self.remaining, self.index = self.budget, 0
        if self.base_reward == "window_pa":
            # Hindsight scoring reference only. Include all assigned bars even
            # when the parent finishes early. No policy or observation gets this mean.
            prices = self.inner.backtest_data.get_deal_price().loc[bars].to_numpy(dtype=float)
            if not len(prices) or not (np.isfinite(prices) & (prices > 0)).all():
                raise ValueError(f"Invalid execution window starting at {self.macro_start}")
            self.window_mean = float(prices.mean())
            self.uniform_value += self.budget * self.window_mean
            uniform_slice = self.budget / self.total
            signed_bps = -self.inner.order.sign * 10_000 / self.inner.twap_price
            self.uniform_bar_rewards = (uniform_slice / self.inner.order.amount) * (prices - self.window_mean) * signed_bps
            self.inventory_phi = 0.0
            self.window_bucket = self.macro_start.strftime("%H%M")
            self.window_stats.setdefault(self.window_bucket, dict(eligible=0, changed=0, delta_pa=0.0))

    def done(self):
        return self.inner.done()

    def _can_adjust(self):
        return (not self.done() and self.enabled and self.scale > 0
                and self.remaining > 1e-8 and self.index < self.total - 1)

    def _arrival_transition(self):
        # Each inner step executes one bar. Fills use actual trade prices.
        # Unexecuted inventory is marked at the bar that just closed, including zero-fill bars.
        bar = self.inner.history_exec.iloc[-1]
        order = self.inner.order
        quantity = float(bar["deal_amount"])
        if not np.isfinite(quantity) or quantity < 0:
            raise ValueError(f"Invalid fill for {order.stock_id}: {quantity}")
        signed_bps = -order.sign * 10_000 / self.arrival_price
        raw = 0.0
        if quantity:
            trade_price = float(bar["trade_price"])
            if not np.isfinite(trade_price) or trade_price <= 0:
                raise ValueError(f"Invalid fill price for {order.stock_id}: {trade_price}")
            raw = (quantity / order.amount) * (trade_price - self.arrival_price) * signed_bps
        if self.shaping_mode == "inventory_mtm":
            market_price = float(bar["market_price"])
            if not np.isfinite(market_price) or market_price <= 0:
                raise ValueError(f"Invalid completed-bar price for {order.stock_id}: {market_price}")
            remaining_fraction = self.inner.position / order.amount
            self.inventory_phi = (0.0 if self.done() else self.coefficient * remaining_fraction
                                  * (market_price - self.arrival_price) * signed_bps)
        return raw

    def _window_transition(self, boundary):
        """Actual fill vs uniform execution of this same macro budget, in parent-order PA bp."""
        bar = self.inner.history_exec.iloc[-1]
        order = self.inner.order
        quantity = float(bar["deal_amount"])
        price = float(bar["market_price"])  # one exogenous 5-minute bar in Simple
        if not np.isfinite(quantity) or quantity < 0 or not np.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid bar for {order.stock_id}: {quantity} at {price}")
        signed_bps = -order.sign * 10_000 / self.inner.twap_price
        edge = (price - self.window_mean) * signed_bps
        # Subtract the uniform fill's contribution each step. Neutral actions
        # from the start of the window now give zero even before summing bars.
        raw = -float(self.uniform_bar_rewards[self.index - 1])
        if quantity:
            fill_price = float(bar["trade_price"])
            if not np.isfinite(fill_price) or fill_price <= 0:
                raise ValueError(f"Invalid trade price for {order.stock_id}: {fill_price}")
            raw += quantity / order.amount * (fill_price - self.window_mean) * signed_bps
        if boundary and self.index < self.total:
            # A finished parent emits no more steps. Debit the neutral benchmark's
            # unvisited tail here, rather than truncating it to the learned finish.
            raw -= float(self.uniform_bar_rewards[self.index:].sum())
        if self.shaping_mode == "relative_inventory_mtm":
            # Mark only the inventory difference from uniform progress. A neutral
            # execution has no potential. Window boundaries (including early
            # terminal fills) close the potential; potentials telescope per window.
            uniform_left = self.budget * (self.total - self.index) / self.total
            relative_inventory = (self.remaining - uniform_left) / order.amount
            self.inventory_phi = 0.0 if boundary else self.coefficient * relative_inventory * edge
        self.raw_total += raw
        self.window_stats[self.window_bucket]["delta_pa"] += raw
        return raw

    def step(self, action):
        if self.done():
            raise RuntimeError("Cannot step a finished order")
        if not 0 <= int(action) < len(self.values):
            raise ValueError("Invalid residual action")
        price_shaping = self.shaping_mode in ("inventory_mtm", "relative_inventory_mtm")
        before = self.inventory_phi if price_shaping else potential(
            self.budget, self.remaining, self.index, self.total, self.coefficient)
        delta = self.values[int(action)] if self.enabled else 0.0
        amount = residual_amount(self.budget, self.remaining, self.index, self.total, delta, self.scale)
        if self._can_adjust():
            self.residual_decisions += 1
            if self.base_reward == "window_pa":
                self.window_stats[self.window_bucket]["eligible"] += 1
            if delta != 0:
                self.nonzero_residual_actions += 1
                reference = residual_amount(self.budget, self.remaining, self.index, self.total, 0.0, self.scale)
                if abs(amount - reference) > max(1e-8, 1e-12 * self.budget):
                    self.changed_residual_amounts += 1
                    if self.base_reward == "window_pa":
                        self.window_stats[self.window_bucket]["changed"] += 1
        old_position = self.inner.position
        self.inner.step(amount)
        self.remaining = max(0.0, self.remaining - (old_position - self.inner.position))
        self.index += 1
        boundary = self.index == self.total or self.done()
        if self.base_reward == "window_pa":
            raw = self._window_transition(boundary)
        elif self.base_reward == "arrival_bps":
            raw = self._arrival_transition()
        else:
            raw = 0.0
        if price_shaping:
            after = self.inventory_phi
        elif boundary:
            after = 0.0
        else:
            after = potential(self.budget, self.remaining, self.index, self.total, self.coefficient)
        if boundary:
            rows = self.inner.history_exec.loc[self.macro_start:]
            metric = self.inner._metrics_collect(self.macro_start, rows.market_volume, rows.market_price,
                                                  self.budget, rows.deal_amount)
            self.macro_history = self.inner._dataframe_append(self.macro_history, [metric])
            self.macro_step += 1
            if self.original_reward is not None:
                raw = float(self.original_reward.reward(self._macro_state()))
            if not self.done():
                if self.macro_step >= self.max_step:
                    raise ValueError("Order horizon exceeds the frozen high-level policy")
                self._start_macro()
        self.transition = raw, before, after, boundary
        if self.done():
            rows = self.inner.history_exec
            self.inner.metrics = self.inner._metrics_collect(self.inner.ticks_index[0], rows.market_volume,
                rows.market_price, self.macro_history.amount.sum(), rows.deal_amount)
            if self.base_reward == "window_pa":
                uniform_price = self.uniform_value / self.inner.order.amount
                uniform_pa = price_advantage(uniform_price, self.inner.twap_price, self.inner.order.direction)
                delta_pa = self.inner.metrics["pa"] - uniform_pa
                if not np.isclose(self.inner.metrics["ffr"], 1., atol=1e-8, rtol=0):
                    raise ValueError("Matched window rewards require a completely filled parent order")
                if not np.isclose(self.raw_total, delta_pa, rtol=1e-9, atol=1e-5):
                    raise RuntimeError(f"Window reward {self.raw_total} differs from matched PA {delta_pa}")
                if self.env is not None:
                    self.env.logger.add_scalar("uniform_pa", uniform_pa)
                    self.env.logger.add_scalar("delta_pa", delta_pa)
                    for bucket, stats in self.window_stats.items():
                        self.env.logger.add_scalar(f"eligible_{bucket}", stats["eligible"])
                        self.env.logger.add_scalar(f"changed_{bucket}", stats["changed"])
                        self.env.logger.add_scalar(f"window_delta_pa_{bucket}", stats["delta_pa"])
            if self.env is not None:
                # Terminal-only counts survive the logger's per-order mean reduction.
                self.env.logger.add_scalar("residual_decisions", self.residual_decisions)
                self.env.logger.add_scalar("nonzero_residual_actions", self.nonzero_residual_actions)
                self.env.logger.add_scalar("changed_residual_amounts", self.changed_residual_amounts)
                for key, value in self.inner.metrics.items():
                    if isinstance(value, float):
                        self.env.logger.add_scalar(key, value)
                    else:
                        self.env.logger.add_any(key, value)

    def get_state(self):
        cursor = int(self.inner.ticks_index.searchsorted(self.inner.cur_time))
        history = self.features[max(0, cursor - self.window):cursor]
        market = np.zeros((self.window, 5), dtype=np.float32)
        if len(history):
            market[-len(history):, :4] = history
            market[-len(history):, 4] = 1  # distinguish padding from a real zero feature
        belief = self.beliefs[cursor - 1] if cursor else self.regime.initial
        entropy = -(belief * np.log(np.clip(belief, 1e-12, 1))).sum() / np.log(self.regime.n_states)
        regime_obs = np.append(belief, entropy).astype(np.float32)
        if not self.regime_enabled:
            regime_obs[:] = 0
        q = self.inner.order.amount
        budget_denominator = max(self.budget, 1e-8)
        valid = self._can_adjust()
        private = np.array([self.inner.position / q, self.remaining / budget_denominator, self.budget / q,
                            (self.total - self.index) / self.total, self.index / self.total,
                            max(0, self.max_step - self.macro_step) / self.max_step,
                            float(self.inner.order.direction), self.macro_step / self.max_step], dtype=np.float32)
        observation = dict(market=market, regime=regime_obs, private=private,
                           actor_valid=np.array([float(valid)], dtype=np.float32))
        return ResidualState(self._macro_state(), observation, *self.transition)


class ResidualStateInterpreter(StateInterpreter):
    def __init__(self, window=12, n_states=3):
        self.window, self.n_states = window, n_states

    @property
    def observation_space(self):
        return spaces.Dict({"market": spaces.Box(-np.inf, np.inf, (self.window, 5), dtype=np.float32),
                            "regime": spaces.Box(0, 1, (self.n_states + 1,), dtype=np.float32),
                            "private": spaces.Box(0, 1, (8,), dtype=np.float32),
                            "actor_valid": spaces.Box(0, 1, (1,), dtype=np.float32)})

    def interpret(self, state):
        return state.observation


class ResidualActionInterpreter(ActionInterpreter):
    def __init__(self, n_actions=5):
        self.n_actions = n_actions

    @property
    def action_space(self):
        return spaces.Discrete(self.n_actions)

    def interpret(self, state, action):
        return int(action)


class PotentialShapingReward(Reward):
    def __init__(self, enabled=True, discount_factor=1.0):
        if discount_factor != 1.0:
            raise ValueError("Use gamma=1 so shaping preserves the order's total reward")
        self.enabled = enabled

    def reward(self, state):
        return state.raw_reward + (state.phi_after - state.phi_before if self.enabled else 0.0)


class RemoteHighPolicy:
    """Only observations/actions cross the pipe; CUDA policy stays in the parent."""
    def __init__(self, state_interpreter, action_interpreter, rpc=None):
        self.state_interpreter, self.action_interpreter = state_interpreter, action_interpreter
        self.rpc = rpc

    def __call__(self, state):
        obs = self.state_interpreter.interpret(state)
        action = self.rpc.request("high_policy", obs)
        return float(self.action_interpreter.interpret(state, action))


class ResidualSimulatorFactory:
    def __init__(self, store, simulator_kwargs):
        self.store, self.kwargs = store, simulator_kwargs

    def __call__(self, order):
        features = self.store.regime_features(order.stock_id, order.start_time,
                                              self.kwargs["regime"], self.kwargs["regime_enabled"])
        return ResidualExecutionSimulator(order, self.store.day("backtest", order.stock_id, order.start_time),
                                           market_features=features, **self.kwargs)

    def worker_copy(self):
        kwargs = dict(self.kwargs)
        high = kwargs["high_policy"]
        kwargs["high_policy"] = RemoteHighPolicy(high.state_interpreter, high.action_interpreter)
        return ResidualSimulatorFactory(self.store, kwargs)

    def bind_rpc(self, rpc):
        self.kwargs["high_policy"].rpc = rpc

    def high_policy_predict(self, observation):
        return self.kwargs["high_policy"].predict_observation(observation)

    def high_policy_predict_batch(self, observations):
        return self.kwargs["high_policy"].predict_observations(observations)
