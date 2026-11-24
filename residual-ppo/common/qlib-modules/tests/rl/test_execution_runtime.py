# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU-only runtime contracts; no training, downloads, or market data required."""
import os
from contextlib import contextmanager
import multiprocessing as mp

from gym import spaces
import numpy as np
import pytest

from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import Simulator
from qlib.rl.utils.env_wrapper import EnvWrapper
from qlib.rl.utils.log import LogLevel
from qlib.rl.utils.parallel_runtime import EnvironmentSpec, PersistentExecutionEnv, InferenceBroker


class CounterSimulator(Simulator):
    def __init__(self, initial, offset):
        super().__init__(initial)
        self.key, self.offset, self.tick = initial, offset, 0
        self.value = float(offset)

    def get_state(self):
        return np.array([self.key, self.tick, self.value], dtype=np.float32)

    def step(self, action):
        self.tick += 1
        self.value += action + 1

    def done(self):
        return self.tick >= 1 + self.key % 3


class CounterState(StateInterpreter):
    @property
    def observation_space(self):
        return spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

    def interpret(self, state):
        return state


class CounterAction(ActionInterpreter):
    @property
    def action_space(self):
        return spaces.Discrete(2)

    def interpret(self, state, action):
        return int(action)


class CounterReward(Reward):
    def reward(self, state):
        return float(state[2])


class CounterFactory:
    def bind_rpc(self, rpc):
        self.rpc = rpc

    def __call__(self, seed):
        if seed < 0:
            raise ValueError("invalid counter seed")
        return CounterSimulator(seed, self.rpc.request("high_policy", seed))


def make_env(high_predict, **options):
    spec = EnvironmentSpec(CounterFactory(), CounterState(), CounterAction(), CounterReward(), LogLevel.PERIODIC)
    return PersistentExecutionEnv(spec, 2, [], high_predict=high_predict, timeout=120, **options)


def collect(env, seeds):
    env.begin_collect(iter(seeds))
    transitions = {}
    with env.collector_guard():
        env.reset()
        while True:
            alive = set(env._alive_env_ids)
            obs, rewards, done, _ = env.step(np.ones(env.env_num, dtype=int))
            for index in alive:
                key, tick, value = obs[index]
                transitions[int(key), int(tick)] = (float(value), float(rewards[index]), bool(done[index]))
            reset_ids = [index for index in alive if done[index]]
            if reset_ids:
                env.reset(reset_ids)
    return transitions


def serial_collect(seeds):
    env = EnvWrapper(lambda seed: CounterSimulator(seed, seed * 2), CounterState(),
                     CounterAction(), iter(seeds), CounterReward())
    transitions = {}
    try:
        for _ in seeds:
            env.reset()
            while True:
                obs, reward, done, _ = env.step(1)
                key, tick, value = obs
                transitions[int(key), int(tick)] = (float(value), float(reward), bool(done))
                if done:
                    break
    finally:
        env.close()
    return transitions


@pytest.mark.parametrize("batch_size", [1, 8])
def test_parallel_rollouts_match_serial_and_reuse_workers(batch_size):
    parent_pid = os.getpid()
    calls = []

    def predict(seed):
        assert os.getpid() == parent_pid
        calls.append(seed)
        return seed * 2

    def predict_batch(seeds):
        return [predict(seed) for seed in seeds]

    env = make_env(predict, high_predict_batch=predict_batch, high_batch_size=batch_size)
    processes = [worker.process for worker in env.workers]
    try:
        pids = env.get_env_attr("worker_pid")
        assert len(set(pids)) == 2 and parent_pid not in pids
        for seeds in ([0, 1, 2, 3, 4], [10], [6, 7, 8, 9]):
            assert collect(env, seeds) == serial_collect(seeds)
            assert env.get_env_attr("worker_pid") == pids
        assert sorted(calls) == list(range(5)) + list(range(6, 11))
    finally:
        env.close()
    assert all(not process.is_alive() for process in processes)


class BrokerWorker:
    """Pipe-only harness: deterministic ready requests without scheduling races."""
    def __init__(self, connection):
        self.connection, self.pending, self.process = connection, True, self
        self.pid, self.exitcode = os.getpid(), None

    def is_alive(self):
        return True


@contextmanager
def broker_connections(broker, count):
    pairs = [mp.Pipe() for _ in range(count)]
    broker.workers = [BrokerWorker(parent) for parent, _ in pairs]
    try:
        yield [child for _, child in pairs]
    finally:
        for parent, child in pairs:
            parent.close()
            child.close()


@pytest.mark.parametrize("batch_size,expected", [(1, [1] * 5), (2, [2, 2, 1]), (8, [5])])
def test_ready_high_requests_are_batched_and_routed(batch_size, expected):
    calls = []

    def batch_predict(seeds):
        calls.append(len(seeds))
        return [seed * 7 + 1 for seed in seeds]

    broker = InferenceBroker(lambda seed: batch_predict([seed])[0], high_predict_batch=batch_predict,
                             high_batch_size=batch_size, high_batch_wait_ms=0, profile_execution=True)
    with broker_connections(broker, 5) as clients:
        for index, client in enumerate(clients):
            client.send(("request", "high_policy", 10 - index))
        broker.pump(timeout=0)
        assert [client.recv() for client in clients] == [("rpc_result", (10 - i) * 7 + 1) for i in range(5)]
    assert calls == expected
    assert broker.profile["requests"] == 5
    assert broker.profile["forwards"] == len(expected)


def test_single_request_does_not_wait_for_other_workers():
    broker = InferenceBroker(lambda seed: seed + 1, high_predict_batch=lambda seeds: [seed + 1 for seed in seeds],
                             high_batch_size=8, high_batch_wait_ms=0.5)
    with broker_connections(broker, 3) as clients:
        # A second worker has already finished; a third has not requested high
        # inference. Neither must be required to complete a batch of eight.
        clients[0].send(("request", "high_policy", 4))
        clients[1].send(("result", "finished"))
        broker.pump(timeout=0)
        assert clients[0].recv() == ("rpc_result", 5)
        assert broker.results[broker.workers[1]] == "finished"


def test_bad_batch_result_reaches_all_waiting_workers():
    broker = InferenceBroker(lambda seed: seed, high_predict_batch=lambda seeds: [], high_batch_wait_ms=0)
    with broker_connections(broker, 2) as clients:
        for client in clients:
            client.send(("request", "high_policy", 1))
        with pytest.raises(RuntimeError, match="wrong number of actions"):
            broker.pump(timeout=0)
        for client in clients:
            kind, error = client.recv()
            assert kind == "rpc_error" and "wrong number of actions" in error


def test_worker_failure_reaches_parent_and_closes_processes():
    env = make_env(lambda seed: seed * 2)
    processes = [worker.process for worker in env.workers]
    try:
        env.begin_collect(iter([-1, 2]))
        with pytest.raises(RuntimeError, match="invalid counter seed"):
            with env.collector_guard():
                env.reset()
    finally:
        env.close()
    assert all(not process.is_alive() for process in processes)
