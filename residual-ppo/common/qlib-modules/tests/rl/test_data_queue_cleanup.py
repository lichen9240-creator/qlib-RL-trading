# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Regression coverage for queue destruction after normal cleanup/shutdown."""
from types import SimpleNamespace
from unittest.mock import Mock

import qlib.rl.utils.data_queue as data_queue_module
from qlib.rl.utils.data_queue import DataQueue


def test_cleanup_after_context_and_closed_handles():
    with DataQueue([1, 2], shuffle=False, queue_maxsize=2) as queue:
        assert list(queue) == [1, 2]
    done_after_context = queue.done()
    queue._queue.close()
    queue._queue.join_thread()

    # Simulate the order seen at shutdown: context cleanup, multiprocessing
    # handle finalization, then Python object destruction.
    queue.cleanup()
    queue.__del__()
    assert queue.done() == done_after_context


def test_destructor_tolerates_incomplete_initialization():
    queue = DataQueue.__new__(DataQueue)
    queue.__del__()


def test_destructor_does_not_use_handles_during_finalization(monkeypatch):
    queue = DataQueue.__new__(DataQueue)
    queue._activated = True
    queue._cleanup_complete = False
    queue.cleanup = Mock(side_effect=AssertionError("shutdown must not access queue handles"))
    monkeypatch.setattr(data_queue_module, "sys", SimpleNamespace(is_finalizing=lambda: True))
    try:
        queue.__del__()
        queue.cleanup.assert_not_called()
    finally:
        queue._cleanup_complete = True
