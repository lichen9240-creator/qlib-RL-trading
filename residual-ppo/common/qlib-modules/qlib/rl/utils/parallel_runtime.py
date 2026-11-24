# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Persistent, synchronous CPU environments with main-process policy inference.

The existing Qlib Collector, replay buffer, PPO update and metric aggregation are
reused. Workers receive neither a Trainer nor a CUDA model. A small RPC broker
serves order seeds and frozen high-level actions while environment steps run in
parallel. Ready high-level requests may share one inference batch; the original
batch-size-one path remains selectable.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback
import warnings

import cloudpickle
from tianshou.env.worker import EnvWorker

from qlib.rl.trainer.trainer import Trainer
from qlib.rl.utils.env_wrapper import EnvWrapper
from qlib.rl.utils.finite_env import FiniteVectorEnv
from qlib.rl.utils.log import LogCollector


class WorkerRPC:
    def __init__(self, connection):
        self.connection = connection

    def request(self, name, payload=None):
        self.connection.send(("request", name, payload))
        kind, result = self.connection.recv()
        if kind == "rpc_error":
            raise RuntimeError(result)
        if kind != "rpc_result":
            raise RuntimeError(f"Unexpected parent response: {kind}")
        return result


class RemoteSeeds:
    def __init__(self, rpc):
        self.rpc = rpc

    def __iter__(self):
        return self

    def __next__(self):
        available, seed = self.rpc.request("next_order")
        if not available:
            raise StopIteration
        return seed


@dataclass
class EnvironmentSpec:
    simulator_fn: object
    state_interpreter: object
    action_interpreter: object
    reward: object
    loglevel: object
    worker_threads: int = 1
    initializer: object = None


def _environment_worker(connection, serialized_spec):
    env = None
    try:
        spec = cloudpickle.loads(serialized_spec)
        import torch

        torch.set_num_threads(spec.worker_threads)
        try:
            import pyarrow
            pyarrow.set_cpu_count(spec.worker_threads)
            if hasattr(pyarrow, "set_io_thread_count"):
                pyarrow.set_io_thread_count(spec.worker_threads)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=r"np\.find_common_type is deprecated\.",
                                category=DeprecationWarning, module=r"pandas\.core\.dtypes\.cast")
        if spec.initializer is not None:
            spec.initializer()
        rpc = WorkerRPC(connection)
        if hasattr(spec.simulator_fn, "bind_rpc"):
            spec.simulator_fn.bind_rpc(rpc)
        env = EnvWrapper(spec.simulator_fn, spec.state_interpreter, spec.action_interpreter,
                         RemoteSeeds(rpc), spec.reward, logger=LogCollector(min_loglevel=spec.loglevel))
        env.worker_pid = os.getpid()
        while True:
            command, value = connection.recv()
            if command == "step":
                result = env.step(value)
            elif command == "reset":
                result = env.reset(**value)
            elif command == "begin_collect":
                env.seed_iterator = RemoteSeeds(rpc)
                env.status = None
                if hasattr(env, "simulator"):
                    del env.simulator
                result = None
            elif command == "getattr":
                result = getattr(env, value, None)
            elif command == "setattr":
                setattr(env.unwrapped, value[0], value[1])
                result = None
            elif command == "seed":
                result = env.seed(value) if hasattr(env, "seed") else None
            elif command == "render":
                result = env.render(**value)
            elif command == "close":
                connection.send(("result", None))
                break
            else:
                raise ValueError(f"Unknown environment command: {command}")
            connection.send(("result", result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            connection.close()


class InferenceBroker:
    def __init__(self, high_predict=None, timeout=120, *, high_predict_batch=None,
                 high_batch_size=8, high_batch_wait_ms=0.5, profile_execution=False):
        if type(high_batch_size) is not int or high_batch_size < 1:
            raise ValueError("high_batch_size must be a positive integer")
        if not math.isfinite(high_batch_wait_ms) or high_batch_wait_ms < 0:
            raise ValueError("high_batch_wait_ms must be finite and nonnegative")
        if type(profile_execution) is not bool:
            raise ValueError("profile_execution must be a boolean")
        self.high_predict, self.timeout = high_predict, timeout
        self.high_predict_batch = high_predict_batch
        self.high_batch_size = high_batch_size
        self.high_batch_wait = high_batch_wait_ms / 1000
        self.profile_execution = profile_execution
        self.workers, self.results = [], {}
        self.seeds = iter(())
        self.reset_profile()

    def reset_profile(self):
        self.profile = dict(requests=0, forwards=0, largest_batch=0,
                            inference_s=0.0, coalesce_wait_s=0.0)

    def report_profile(self):
        if self.profile_execution and self.profile["requests"]:
            p = self.profile
            print(f"[High inference] requests={p['requests']}, forwards={p['forwards']}, "
                  f"mean_batch={p['requests'] / p['forwards']:.2f}, max_batch={p['largest_batch']}, "
                  f"inference={p['inference_s']:.3f}s, coalesce_wait={p['coalesce_wait_s']:.3f}s", flush=True)

    @staticmethod
    def _rpc_error(requests, error):
        for connection, _ in requests:
            try:
                connection.send(("rpc_error", error))
            except (EOFError, BrokenPipeError, OSError):
                pass

    def _predict(self, requests):
        started = time.perf_counter() if self.profile_execution else None
        if self.high_predict_batch is not None and self.high_batch_size > 1:
            actions = list(self.high_predict_batch([payload for _, payload in requests]))
        else:
            actions = [self.high_predict(payload) for _, payload in requests]
        if len(actions) != len(requests):
            raise RuntimeError("High-level inference returned the wrong number of actions")
        if self.profile_execution:
            self.profile["requests"] += len(requests)
            self.profile["forwards"] += 1
            self.profile["largest_batch"] = max(self.profile["largest_batch"], len(requests))
            self.profile["inference_s"] += time.perf_counter() - started
        for (connection, _), action in zip(requests, actions):
            connection.send(("rpc_result", action))

    def _read_message(self, connection, worker, high_requests):
        try:
            message = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Execution worker {worker.process.pid} exited unexpectedly") from exc
        kind = message[0]
        if kind == "result":
            self.results[worker] = message[1]
        elif kind == "error":
            raise RuntimeError(f"Execution worker {worker.process.pid}:\n{message[1]}")
        elif kind == "request":
            _, name, payload = message
            try:
                if name == "next_order":
                    try:
                        result = (True, next(self.seeds))
                    except StopIteration:
                        result = (False, None)
                    connection.send(("rpc_result", result))
                elif name == "high_policy" and self.high_predict is not None:
                    if self.high_predict_batch is not None and self.high_batch_size > 1:
                        high_requests.append((connection, payload))
                    else:
                        self._predict([(connection, payload)])
                else:
                    raise ValueError(f"Unsupported worker RPC: {name}")
            except BaseException:
                self._rpc_error([(connection, payload)], traceback.format_exc())
                raise
        else:
            raise RuntimeError(f"Unexpected worker message: {kind}")

    def pump(self, timeout=0.1):
        pending = [w for w in self.workers if w.pending and w not in self.results]
        if not pending:
            return
        connections = {w.connection: w for w in pending}
        ready = wait(list(connections), timeout=timeout)
        high_requests = []
        try:
            for connection in ready:
                self._read_message(connection, connections[connection], high_requests)
            # Coalesce only this round's pending connections, with a bounded
            # window. Never wait for idle, exhausted, or already-returned workers.
            # A singleton request must be served even if no other worker calls
            # the high-level policy on this micro step.
            deadline = time.monotonic() + self.high_batch_wait
            while high_requests and len(high_requests) < self.high_batch_size:
                queued = {connection for connection, _ in high_requests}
                available = [connection for connection, worker in connections.items()
                             if connection not in queued and worker not in self.results]
                if not available:
                    break
                remaining = max(0.0, deadline - time.monotonic())
                started = time.perf_counter() if self.profile_execution else None
                more = wait(available, timeout=remaining)
                if self.profile_execution:
                    self.profile["coalesce_wait_s"] += time.perf_counter() - started
                for connection in more:
                    self._read_message(connection, connections[connection], high_requests)
                if not more or time.monotonic() >= deadline:
                    break
            while high_requests:
                batch = high_requests[:self.high_batch_size]
                self._predict(batch)
                del high_requests[:len(batch)]
        except BaseException:
            self._rpc_error(high_requests, traceback.format_exc())
            raise
        for worker in pending:
            if worker not in self.results and not worker.process.is_alive() and not worker.connection.poll():
                raise RuntimeError(f"Execution worker {worker.process.pid} died (exit={worker.process.exitcode})")

    def receive(self, worker, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while worker not in self.results:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Execution worker {worker.process.pid} did not respond")
            self.pump()
        worker.pending = False
        return self.results.pop(worker)

    def close(self):
        for worker in list(self.workers):
            worker.close()


class SpawnExecutionWorker(EnvWorker):
    def __init__(self, spec, broker):
        self.broker, self.pending, self.is_closed = broker, False, False
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_environment_worker, args=(child, cloudpickle.dumps(spec)), daemon=True)
        self.process.start()
        child.close()
        broker.workers.append(self)
        super().__init__(spec)

    def command(self, name, value=None):
        if self.pending:
            raise RuntimeError("Cannot send a second command to a busy execution worker")
        self.pending = True
        self.connection.send((name, value))

    def send(self, action, **kwargs):
        self.command("reset", kwargs) if action is None else self.command("step", action)

    def recv(self):
        return self.broker.receive(self)

    def reset(self, **kwargs):
        self.send(None, **kwargs)
        return self.recv()

    def get_env_attr(self, key):
        self.command("getattr", key)
        return self.recv()

    def set_env_attr(self, key, value):
        self.command("setattr", (key, value))
        return self.recv()

    def seed(self, seed=None):
        super().seed(seed)
        self.command("seed", seed)
        return self.recv()

    def render(self, **kwargs):
        self.command("render", kwargs)
        return self.recv()

    @staticmethod
    def wait(workers, wait_num, timeout=None):
        if not workers:
            return []
        broker, started = workers[0].broker, time.monotonic()
        while True:
            ready = [w for w in workers if w in broker.results]
            if len(ready) >= wait_num or (timeout is not None and time.monotonic() - started >= timeout):
                return ready
            broker.pump()

    def close_env(self):
        try:
            if self.process.is_alive() and not self.pending:
                self.command("close")
                self.broker.receive(self, timeout=3)
        except (OSError, RuntimeError, TimeoutError):
            pass
        finally:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.connection.close()
            self.broker.results.pop(self, None)
            if self in self.broker.workers:
                self.broker.workers.remove(self)


class PersistentExecutionEnv(FiniteVectorEnv):
    def __init__(self, spec, concurrency, loggers, high_predict=None, timeout=120, **inference_options):
        self.broker = InferenceBroker(high_predict, timeout, **inference_options)
        try:
            super().__init__(loggers, [spec] * concurrency,
                             worker_fn=lambda item: SpawnExecutionWorker(item, self.broker))
        except BaseException:
            self.broker.close()
            raise

    def begin_collect(self, iterator):
        if any(w.pending for w in self.workers):
            raise RuntimeError("Previous collection still has pending environment commands")
        self.broker.seeds = iter(iterator)
        for worker in self.workers:
            worker.command("begin_collect")
        for worker in self.workers:
            worker.recv()
        self._alive_env_ids = set(range(self.env_num))
        self._zombie = False
        self._default_obs = self._default_info = self._default_rew = None

    @contextmanager
    def collector_guard(self):
        self.broker.reset_profile()
        try:
            with super().collector_guard():
                yield self
        finally:
            self.broker.report_profile()

    def close(self):
        if not self.is_closed:
            self.broker.close()
            self.is_closed = True


class ParallelExecutionTrainer(Trainer):
    """Override environment placement only; keep upstream fit/train/test loops."""
    def __init__(self, *, execution_mode="subproc", worker_threads=1, worker_timeout=120,
                 worker_initializer=None, high_batch_size=8, high_batch_wait_ms=0.5,
                 profile_execution=False, **kwargs):
        super().__init__(**kwargs)
        if execution_mode not in ("dummy", "subproc"):
            raise ValueError("execution_mode must be dummy or subproc")
        self.execution_mode = execution_mode
        if type(worker_threads) is not int or worker_threads < 1 or worker_timeout <= 0:
            raise ValueError("worker_threads must be a positive integer and worker_timeout must be positive")
        self.worker_threads, self.worker_timeout = worker_threads, worker_timeout
        self.worker_initializer = worker_initializer
        self.inference_options = dict(high_batch_size=high_batch_size, high_batch_wait_ms=high_batch_wait_ms,
                                      profile_execution=profile_execution)
        self._execution_env = None

    def venv_from_iterator(self, iterator):
        if self.execution_mode == "dummy":
            # The existing serial implementation remains available for parity checks.
            self.finite_env_type = "dummy"
            return super().venv_from_iterator(iterator)
        if self._execution_env is None:
            factory = self.vessel.simulator_fn
            high_predict = getattr(factory, "high_policy_predict", None)
            worker_factory = factory.worker_copy() if hasattr(factory, "worker_copy") else factory
            spec = EnvironmentSpec(worker_factory, copy.deepcopy(self.vessel.state_interpreter),
                copy.deepcopy(self.vessel.action_interpreter), copy.deepcopy(self.vessel.reward),
                self._min_loglevel(), self.worker_threads, self.worker_initializer)
            self._execution_env = PersistentExecutionEnv(spec, self.concurrency, self.loggers,
                high_predict, self.worker_timeout, high_predict_batch=getattr(factory, "high_policy_predict_batch", None),
                **self.inference_options)
            print(f"[Execution workers] spawn, {self.concurrency} persistent CPU processes; "
                  f"PIDs={[w.process.pid for w in self._execution_env.workers]}", flush=True)
        self._execution_env.begin_collect(iterator)
        return self._execution_env

    def fit(self, vessel, ckpt_path=None):
        try:
            return super().fit(vessel, ckpt_path)
        finally:
            self.close_workers()

    def test(self, vessel):
        try:
            return super().test(vessel)
        finally:
            self.close_workers()

    def close_workers(self):
        if self._execution_env is not None:
            self._execution_env.close()
            self._execution_env = None
