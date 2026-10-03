import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from ale_py.vector_env import AtariVectorEnv

from . import logger


def _load(policy, slot, task):
    if policy.seeds[slot] != task.seeds:
        policy.load(slot, task.make_theta(), task.seeds)


@dataclass
class Task:
    """One episode to play: ``make_theta`` lazily builds the flat weight vector on demand,
    so a population of 1000 genomes never has to be materialized at once."""
    seeds: tuple
    make_theta: Callable[[], torch.Tensor]


class AtariEvaluator:
    """Plays episodes for many genomes in parallel.

    Every sub-environment of an ale-py ``AtariVectorEnv`` is a *slot* with its own weight
    set inside ``BatchedPolicy``. When an episode ends (game over, or the per-phase step
    limit), its slot is loaded with the next pending task and only that sub-environment
    starts a new episode. Environment preprocessing matches
    the original gym_tensorflow setup: 4-frame skip with max-pooling, 84x84 grayscale,
    4 stacked frames, 1-30 random no-ops, no sticky actions and unclipped rewards.
    """

    def __init__(self, game, num_envs, env_kwargs=None, num_threads=0, seed=None):
        validate_env_kwargs(env_kwargs or {})
        kwargs = dict(
            frameskip=4,
            stack_num=4,
            img_height=84,
            img_width=84,
            grayscale=True,
            maxpool=True,
            noop_max=30,
            repeat_action_probability=0.0,
            reward_clipping=False,
            use_fire_reset=False,
            episodic_life=False,
            max_num_frames_per_episode=108_000,
        )
        kwargs.update(env_kwargs or {})
        self.game = game
        self.num_envs = num_envs
        # SameStep: a sub-environment that terminates is reset inside the same step() call,
        # so the returned observation already belongs to the next episode of that slot.
        self.env = AtariVectorEnv(game, num_envs, num_threads=num_threads, autoreset_mode="SameStep", **kwargs)
        self.num_actions = int(self.env.single_action_space.n)
        self.obs_shape = tuple(self.env.single_observation_space.shape)
        self.steps_counter = 0
        self.random = np.random.RandomState(seed)

    def close(self):
        self.env.close()

    def run(self, policy, tasks, max_steps=None, log_interval=10.0):
        """Play one episode per task. Returns ``[(seeds, episode_reward, episode_length)]`` in task order."""
        assert policy.num_slots == self.num_envs
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        num_tasks = len(tasks)
        if not num_tasks:
            return []
        results = [None] * num_tasks
        slot_task = np.full(self.num_envs, -1)
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        lengths = np.zeros(self.num_envs, dtype=np.int64)
        next_task = 0

        for slot in range(min(self.num_envs, num_tasks)):
            _load(policy, slot, tasks[next_task])
            slot_task[slot] = next_task
            next_task += 1

        # Seed each phase independently so a generation-boundary checkpoint needs
        # only this RNG state, rather than serializing the live C++ emulators.
        obs, _ = self.env.reset(seed=int(self.random.randint(0, 2**31 - self.num_envs)))
        finished = 0
        t_start = t_log = time.time()
        steps_at_log = self.steps_counter

        while finished < num_tasks:
            running = slot_task >= 0
            actions = policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, rew, terminated, truncated, _ = self.env.step(actions)
            rewards[running] += rew[running]
            lengths[running] += 1
            self.steps_counter += int(running.sum())

            env_done = terminated | truncated
            cut_off = running & ~env_done & (lengths >= max_steps) if max_steps is not None \
                else np.zeros_like(env_done)
            done = (env_done & running) | cut_off

            for slot in np.flatnonzero(done):
                task_id = slot_task[slot]
                results[task_id] = (tasks[task_id].seeds, float(rewards[slot]), int(lengths[slot]))
                finished += 1
                rewards[slot] = 0.0
                lengths[slot] = 0
                if next_task < num_tasks:
                    _load(policy, slot, tasks[next_task])
                    slot_task[slot] = next_task
                    next_task += 1
                else:
                    slot_task[slot] = -1
            if cut_off.any():
                obs, _ = self.env.reset(options={"reset_mask": cut_off})

            if time.time() - t_log > log_interval:
                rate = (self.steps_counter - steps_at_log) / (time.time() - t_log)
                logger.info(f"  {finished}/{num_tasks} episodes done, {rate:.0f} steps/s")
                t_log, steps_at_log = time.time(), self.steps_counter

        logger.info(f"Done evaluating {num_tasks} episodes in {time.time() - t_start:.2f} seconds")
        return results

    def run_repeated(self, policy, tasks, num_episodes, max_steps=None):
        """Play ``num_episodes`` per task. Returns ``[(seeds, rewards[], lengths[])]`` in task order."""
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        expanded = [task for task in tasks for _ in range(num_episodes)]
        flat = self.run(policy, expanded, max_steps=max_steps)
        grouped = []
        for i in range(len(tasks)):
            chunk = flat[i * num_episodes:(i + 1) * num_episodes]
            grouped.append((chunk[0][0], np.array([c[1] for c in chunk]), np.array([c[2] for c in chunk])))
        return grouped


def validate_env_kwargs(kwargs):
    """Keep slot ordering and the original genome's input layout well-defined."""
    fixed = dict(batch_size=0, autoreset_mode="SameStep", grayscale=True,
                 continuous=False, stack_num=4, img_height=84, img_width=84, frameskip=4)
    reserved = {"game", "num_envs", "num_threads"}
    for key, value in kwargs.items():
        if key in reserved or (key in fixed and value != fixed[key]):
            raise ValueError(f"Unsupported env_kwargs[{key!r}]: {value!r}")
    # These options are supplied by the evaluator, never forwarded twice.
    if "autoreset_mode" in kwargs:
        raise ValueError("autoreset_mode is managed by AtariEvaluator; omit it from env_kwargs")
