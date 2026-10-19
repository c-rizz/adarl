"""Episode length bookkeeping: do episodes last as long as they are configured to?

There was no env-level test at all, and a total training blocker hid in that gap: in loco_kyon_ppo
every env reported done on *every* step, so 8 million "episodes" of 1.17 steps were collected in an
hour while the robot was standing perfectly still. It went unnoticed for hours because the only
visible symptom was that every per-episode number (reward, episode length, each reward term) was a
single step's worth, which reads as "the policy isn't learning" rather than "the counters are wrong".

These tests drive the real BaseVecEnv/EnvRunner bookkeeping with a minimal env whose only state is
its own step counter, so the arithmetic is checkable exactly: an episode must end on the step where
the counter reaches that env's limit, never earlier, and the counter must go back to zero after.
Everything runs on the CPU, so it is part of the fast lane.
"""
from typing import Mapping

import numpy as np
import pytest
import torch as th

from adarl.envs.vec.BaseVecEnv import BaseVecEnv
from adarl.envs.vec.EnvRunner import EnvRunner
import adarl.utils.spaces as spaces


class CounterVecEnv(BaseVecEnv[Mapping[str, th.Tensor]]):
    """A vec env with no physics: its state is the number of steps since the episode started.

    Truncation is the usual `step counter >= this env's limit`, which is what every real env in the
    workspace does, and nothing ever terminates, so any episode that ends early or late is a
    bookkeeping bug rather than something the dynamics did.
    """

    def __init__(self, num_envs: int, max_ep_steps: th.Tensor, th_device=th.device("cpu")):
        box = spaces.gym_spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.float32)
        super().__init__(num_envs=num_envs,
                         single_action_space=spaces.gym_spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32),
                         single_observation_space=spaces.gym_spaces.Dict({"count": box}),
                         single_state_space=spaces.gym_spaces.Dict({"count": box}),
                         info_space=spaces.gym_spaces.Dict({}),
                         th_device=th_device,
                         max_episode_steps=max_ep_steps,
                         max_possible_episode_steps=int(max_ep_steps.max().item()))
        self.ep_lengths_seen: list[th.Tensor] = []

    # --- the bits the runner drives -------------------------------------------------------------
    def _build(self):
        pass

    def _initialize_episodes(self, vec_mask=None, options={}):
        pass  # BaseVecEnv already zeroed the step counter for the envs in vec_mask

    def submit_actions(self, actions: th.Tensor):
        pass

    # step() is deliberately not overridden: BaseVecEnv.step() is what advances _ep_step_counter,
    # which is the bookkeeping under test

    def get_states(self):
        return {"count": self._ep_step_counter.reshape(self.num_envs, 1).to(th.float32)}

    def get_observations(self, states):
        return states

    def get_times_since_build(self):
        return self._ep_step_counter.to(th.float32)

    def get_times_since_ep_start(self):
        return self._ep_step_counter.to(th.float32)

    def are_states_timedout(self, states):
        return self._ep_step_counter >= self._max_ep_steps

    def are_states_terminal(self, states):
        return th.zeros((self.num_envs,), dtype=th.bool, device=self._th_device)

    def compute_rewards(self, states, sub_rewards_return=None):
        reward = th.ones((self.num_envs,), dtype=th.float32, device=self._th_device)
        if sub_rewards_return is not None:
            # the runner stacks the sub-rewards, so it needs at least one named term
            sub_rewards_return["alive"] = reward
        return reward

    def get_ui_renderings(self, vec_mask):
        return [], th.zeros((self.num_envs,), dtype=th.float32, device=self._th_device)

    def get_infos(self, state, labels=None):
        return {}

    def close(self):
        pass


def _run(limits: list[int], steps: int):
    """Drive the runner and record, per env, the step counts at which episodes ended."""
    num_envs = len(limits)
    max_ep_steps = th.as_tensor(limits, dtype=th.long)
    env = CounterVecEnv(num_envs=num_envs, max_ep_steps=max_ep_steps)
    runner = EnvRunner(env, quiet=True, autoreset=True)
    runner.reset()
    action = th.zeros((num_envs, 1), dtype=th.float32)
    ended_at: list[list[int]] = [[] for _ in range(num_envs)]
    # counted here rather than read off the env: with autoreset the env's own counter is already back
    # to zero by the time step() returns, and this is also exactly how a logger measures durations
    live = [0] * num_envs
    for _ in range(steps):
        terminated, truncated = runner.step(action)[3:5]
        done = th.logical_or(th.as_tensor(terminated).reshape(-1).bool(),
                             th.as_tensor(truncated).reshape(-1).bool())
        for e in range(num_envs):
            live[e] += 1
            if bool(done[e]):
                ended_at[e].append(live[e])
                live[e] = 0
    runner.close()
    return ended_at


def test_episodes_end_exactly_at_their_own_limit():
    """Each env must end its episode on the step its own counter reaches its own limit."""
    limits = [3, 5, 8, 11]
    ended_at = _run(limits, steps=40)
    for e, limit in enumerate(limits):
        assert ended_at[e], (f"env {e} never finished an episode in 40 steps although its limit is "
                             f"{limit}: episodes are not being truncated at all")
        wrong = [n for n in ended_at[e] if n != limit]
        assert not wrong, (f"env {e} has a limit of {limit} steps but ended episodes at step counts "
                           f"{ended_at[e]}: {sorted(set(wrong))} are not {limit}. An episode ending at "
                           f"1 every time is the loco_kyon_ppo failure this test exists for.")


def test_step_counter_resets_between_episodes():
    """After an episode ends the counter must restart, or every later step truncates immediately."""
    limits = [4, 4, 4, 4]
    ended_at = _run(limits, steps=30)
    for e in range(len(limits)):
        assert len(ended_at[e]) >= 2, (f"env {e} completed {len(ended_at[e])} episodes in 30 steps with a "
                                       f"4-step limit; expected several")
        assert all(n == 4 for n in ended_at[e]), (
            f"env {e} ended episodes at {ended_at[e]} instead of always at 4: the step counter is not "
            f"going back to zero, so every step after the first episode is past the limit")


def test_mean_episode_length_matches_the_limits():
    """The aggregate a training log would show must match the configured limits.

    This is the check that would have made the 1.17-step episodes obvious: with these limits the mean
    episode length can only be ~6, and no amount of policy behaviour can change that.
    """
    limits = [3, 5, 8, 11]
    steps = 120
    ended_at = _run(limits, steps=steps)
    total_eps = sum(len(x) for x in ended_at)
    assert total_eps > 0, "no episodes completed at all"
    mean_len = steps * len(limits) / total_eps
    expected = float(np.mean(limits))
    assert abs(mean_len - expected) < 0.25 * expected, (
        f"mean episode length {mean_len:.2f} does not match the configured limits {limits} "
        f"(expected about {expected:.2f}): the episode accounting disagrees with the step count")
