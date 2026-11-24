# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""User-run forward/backward and frozen-policy parity checks. No external data."""
import copy

from gym import spaces
import numpy as np
import pytest
import torch
from tianshou.data import Batch

from qlib.rl.order_execution.network import Recurrent
from qlib.rl.order_execution.policy import PPO
from qlib.rl.order_execution.residual_policy import ResidualNetwork, ResidualPPO
from qlib.rl.order_execution.residual_simulator import (
    FrozenHighPolicy, ResidualStateInterpreter, ResidualActionInterpreter,
)


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"))]


def observations(kind, device):
    lengths = [0, 1, 3, 7, 11, 12] if kind != "empty" else [0, 0, 0]
    market = torch.randn(len(lengths), 12, 5, device=device)
    market[:, :, 4] = 0
    for row, length in enumerate(lengths):
        if length:
            market[row, -length:, 4] = 1
    if kind == "gapped":
        market[2, :, 4] = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], device=device)
    # Padding feature values deliberately remain nonzero: they must be ignored.
    return dict(market=market.requires_grad_(),
                private=torch.randn(len(lengths), 8, device=device, requires_grad=True),
                regime=torch.randn(len(lengths), 4, device=device, requires_grad=True))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["mixed", "empty", "gapped"])
def test_sequence_gru_preserves_forward_and_backward(device, kind):
    torch.manual_seed(19)
    reference = ResidualNetwork(hidden_dim=8, private_dim=8, gru_sequence=False).to(device)
    optimized = copy.deepcopy(reference)
    optimized.gru_sequence = True
    left = observations(kind, device)
    right = {key: value.detach().clone().requires_grad_() for key, value in left.items()}
    expected, actual = reference(left), optimized(right)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for result in (expected, actual):
        probe = torch.arange(1, result.numel() + 1, device=device).reshape_as(result) / result.numel()
        (result * probe).sum().backward()
    for key in left:
        torch.testing.assert_close(right[key].grad, left[key].grad, rtol=2e-5, atol=2e-6)
    for (name, old), (new_name, new) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert name == new_name and old.grad is not None and new.grad is not None
        torch.testing.assert_close(new.grad, old.grad, rtol=2e-5, atol=2e-6, msg=name)
        if kind == "empty" and name.startswith("market_gru."):
            assert torch.count_nonzero(new.grad) == 0
    assert torch.count_nonzero(right["market"].grad[left["market"][:, :, 4] == 0]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_policy_checkpoint_logits_values_and_forced_actions_are_compatible(device):
    torch.manual_seed(23)
    state, action = ResidualStateInterpreter(), ResidualActionInterpreter()
    reference = ResidualPPO(state.observation_space, action.action_space, gru_sequence=False).to(device).eval()
    # Avoid a trivial test against the initial, constant zero-action bias.
    with torch.no_grad():
        reference.actor.logits.weight.normal_(0, 0.1)
    optimized = ResidualPPO(state.observation_space, action.action_space, gru_sequence=True).to(device).eval()
    assert set(optimized.state_dict()) == set(reference.state_dict())
    optimized.load_state_dict(reference.state_dict(), strict=True)
    obs = observations("mixed", device)
    obs["actor_valid"] = torch.tensor([[1], [0], [1], [1], [0], [1]], device=device, dtype=torch.float32)
    with torch.no_grad():
        left = reference(Batch(obs=obs, info=Batch()))
        right = optimized(Batch(obs=obs, info=Batch()))
        torch.testing.assert_close(right.logits, left.logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(optimized.critic(obs), reference.critic(obs), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(right.act, left.act, rtol=0, atol=0)
        assert torch.all(right.act[[1, 4]] == 2)


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_high_batch_actions_match_single_requests(device):
    torch.manual_seed(29)
    rng = np.random.default_rng(29)
    obs_space = spaces.Dict({"data_processed": spaces.Box(-np.inf, np.inf, (48, 5), dtype=np.float32)})
    policy = PPO(Recurrent(obs_space, hidden_dim=8, output_dim=8), obs_space, spaces.Discrete(5),
                 lr=1e-4, deterministic_eval=True).to(device)
    high = FrozenHighPolicy(policy, None, None)
    observations_list = []
    for tick, step in ((0, 0), (1, 0), (6, 1), (13, 2), (30, 5), (42, 7), (47, 7)):
        data = rng.normal(size=(48, 5)).astype(np.float32)
        data[tick:] = 0
        observations_list.append(dict(data_processed=data, cur_tick=np.array(tick, dtype=np.int32),
            cur_step=np.array(step, dtype=np.int32), position_history=np.linspace(100, 0, 8, dtype=np.float32),
            target=np.array(100, dtype=np.float32), num_step=np.array(8, dtype=np.int32),
            acquiring=np.array(step % 2, dtype=np.float32)))
    assert high.predict_observations([]) == []
    expected = [high.predict_observation(obs) for obs in observations_list]
    assert high.predict_observations(observations_list) == expected
    assert high.predict_observations(observations_list[:1]) == expected[:1]
    with torch.no_grad():
        single_probs = torch.cat([policy(Batch(obs=Batch([obs]), info=Batch())).logits for obs in observations_list])
        batch_probs = policy(Batch(obs=Batch(observations_list), info=Batch())).logits
    torch.testing.assert_close(batch_probs, single_probs, rtol=2e-5, atol=2e-6)
