# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Micro execution PPO, with forced actions excluded only from actor losses."""
import torch
from torch import nn
from tianshou.data import to_torch
from tianshou.policy import PPOPolicy

from qlib.rl.order_execution.policy import PPOCritic, auto_device, chain_dedup
from qlib.rl.trainer.vessel import TrainingVessel


DEFAULT_ZERO_ACTION_BIAS = 0.01


class ResidualNetwork(nn.Module):
    def __init__(self, n_states=3, hidden_dim=64, private_dim=32, gru_sequence=True):
        super().__init__()
        if type(gru_sequence) is not bool:
            raise ValueError("gru_sequence must be a boolean")
        # Execution choice only: no new parameters/buffers or state_dict keys.
        self.gru_sequence = gru_sequence
        self.output_dim = hidden_dim
        self.market_gru = nn.GRU(5, hidden_dim, batch_first=True)
        self.private = nn.Sequential(nn.Linear(8 + n_states + 1, private_dim), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + private_dim, hidden_dim), nn.Tanh())

    def _market_hidden_loop(self, market):
        """Original reference path, including the zero-gradient padding branches."""
        hidden = torch.zeros(1, market.shape[0], self.output_dim, device=market.device)
        for t in range(market.shape[1]):
            _, candidate = self.market_gru(market[:, t:t + 1], hidden)
            hidden = torch.where(market[:, t, 4][None, :, None] > 0, candidate, hidden)
        return hidden[0]

    def _market_hidden_sequence(self, market):
        """One unidirectional GRU call over the same valid timesteps, in order.

        Move valid entries before padding, then select the last VALID output.
        Do not use h_n: it includes the trailing padding and GRU biases. Keys
        are unique, so this also handles interior masked entries without a
        stable-sort requirement. Lengths/indexing stay on the input device.
        """
        valid = market[:, :, 4] > 0
        length = valid.sum(dim=1)
        width = market.shape[1]
        position = torch.arange(width, device=market.device).unsqueeze(0)
        permutation = torch.where(valid, position, position + width).argsort(dim=1)
        sequence = market.gather(1, permutation.unsqueeze(-1).expand_as(market))
        # Mask unused suffix values (including all-empty histories). They must
        # never affect valid-prefix outputs or input/parameter gradients.
        sequence = torch.where((position < length.unsqueeze(1)).unsqueeze(-1),
                               sequence, torch.zeros_like(sequence))
        output, _ = self.market_gru(sequence)
        last = (length - 1).clamp_min(0)
        hidden = output.gather(1, last[:, None, None].expand(-1, 1, self.output_dim)).squeeze(1)
        # Keep a differentiable zero branch, as in the original torch.where:
        # all-padding batches produce zero GRU gradients, not missing gradients.
        return torch.where(length[:, None] > 0, hidden, torch.zeros_like(hidden))

    def forward(self, obs):
        market = obs["market"].float()
        hidden = (self._market_hidden_sequence(market) if self.gru_sequence
                  else self._market_hidden_loop(market))
        private = self.private(torch.cat((obs["private"].float(), obs["regime"].float()), -1))
        return self.fusion(torch.cat((hidden, private), -1))


class ResidualActor(nn.Module):
    def __init__(self, extractor, action_dim, zero_action, initial_zero_bias=DEFAULT_ZERO_ACTION_BIAS):
        super().__init__()
        self.extractor, self.zero_action = extractor, zero_action
        self.logits = nn.Linear(extractor.output_dim, action_dim)
        # A small initial margin breaks the argmax tie in favor of the reference
        # schedule while giving all five actions nearly equal sampling probability.
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        with torch.no_grad():
            self.logits.bias[zero_action] = initial_zero_bias

    def forward(self, obs, state=None, info=None):
        obs = to_torch(obs, device=auto_device(self))
        logits = self.logits(self.extractor(obs))
        forced = torch.full_like(logits, -1e9)
        forced[:, self.zero_action] = 0
        logits = torch.where(obs["actor_valid"].bool(), logits, forced)
        return logits.softmax(-1), state


def masked_actor_loss(dist, act, old_log_prob, advantages, valid, clip, normalize=True):
    valid = valid.reshape(-1).bool()
    if not valid.any():
        zero = dist.probs.sum() * 0
        return zero, zero
    adv = advantages.reshape(-1)[valid]
    if normalize and len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    ratio = (dist.log_prob(act).reshape(-1)[valid] - old_log_prob.reshape(-1)[valid]).exp()
    loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    return loss, dist.entropy().reshape(-1)[valid].mean()


class ResidualPPO(PPOPolicy):
    """Tianshou 0.4.10 PPO/GAE; critic retains all transitions and macro boundaries."""
    def __init__(self, obs_space, action_space, n_states=3, zero_action=2, hidden_dim=64, private_dim=32,
                 lr=1e-4, discount_factor=1.0, gae_lambda=0.95, eps_clip=0.2, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, initial_zero_bias=DEFAULT_ZERO_ACTION_BIAS, gru_sequence=True):
        if discount_factor != 1.0:
            raise ValueError("discount_factor must be 1 for the undiscounted order reward")
        extractor = ResidualNetwork(n_states, hidden_dim, private_dim, gru_sequence=gru_sequence)
        actor = ResidualActor(extractor, action_space.n, zero_action, initial_zero_bias)
        critic = PPOCritic(extractor)
        optimizer = torch.optim.Adam(chain_dedup(actor.parameters(), critic.parameters()), lr=lr)
        super().__init__(actor, critic, optimizer, torch.distributions.Categorical,
                         discount_factor=discount_factor, gae_lambda=gae_lambda, eps_clip=eps_clip,
                         vf_coef=vf_coef, ent_coef=ent_coef, max_grad_norm=max_grad_norm,
                         reward_normalization=False, value_clip=False, advantage_normalization=True,
                         deterministic_eval=True, observation_space=obs_space, action_space=action_space)

    def learn(self, batch, batch_size, repeat, **kwargs):
        print(f"[PPO update] {len(batch)} transitions, {repeat} passes, batch_size={batch_size}", flush=True)
        logs = {name: [] for name in ("loss", "loss/clip", "loss/vf", "loss/ent")}
        for _ in range(repeat):
            for mini in batch.split(batch_size, merge_last=True):
                dist = self(mini).dist
                valid = torch.as_tensor(mini.obs.actor_valid, device=mini.adv.device)
                actor_loss, entropy = masked_actor_loss(dist, mini.act, mini.logp_old, mini.adv, valid,
                                                        self._eps_clip, self._norm_adv)
                value_loss = (mini.returns - self.critic(mini.obs).flatten()).square().mean()
                loss = actor_loss + self._weight_vf * value_loss - self._weight_ent * entropy
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._grad_norm)
                self.optim.step()
                for key, value in zip(logs, (loss, actor_loss, value_loss, entropy)):
                    logs[key].append(value.item())
        print("[PPO update] Complete", flush=True)
        return logs


class ResidualTrainingVessel(TrainingVessel):
    """Include optimizer and artifact identities in ordinary Qlib checkpoints."""
    def __init__(self, artifact_metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.artifact_metadata = artifact_metadata or {}

    def state_dict(self):
        return dict(super().state_dict(), optimizer=self.policy.optim.state_dict(),
                    artifacts=self.artifact_metadata)

    def load_state_dict(self, state_dict):
        if state_dict.get("artifacts") != self.artifact_metadata:
            raise ValueError("Checkpoint configuration or frozen artifact differs from this run")
        super().load_state_dict(state_dict)
        self.policy.optim.load_state_dict(state_dict["optimizer"])
