"""Regression checks with real ALE episodes, checkpoints, and controlled rewards."""
import json
import os
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ga
from neuroevolution import AtariEvaluator, BatchedPolicy, Task
from neuroevolution.evaluator import validate_env_kwargs
from play import load_genome


def tiny_config(**overrides):
    return {**ga.DEFAULTS, 'game': 'frostbite', 'model': 'Model',
            'population_size': 4, 'selection_threshold': 2, 'validation_threshold': 2,
            'num_validation_episodes': 2, 'num_test_episodes': 2,
            'episode_cutoff_mode': 8, 'test_max_steps': 6, 'timesteps': 128,
            'mutation_power': 0.002, 'num_envs': 2, 'num_threads': 1,
            'noise_table_size': 1_100_000, 'device': 'cpu', 'seed': 42, **overrides}


def read_state(path):
    with open(path / 'snapshot.pkl', 'rb') as f:
        return pickle.load(f)


@pytest.mark.parametrize('selection_threshold', [0, 2])
def test_resume_matches_uninterrupted_run(tmp_path, selection_threshold):
    config = tiny_config(selection_threshold=selection_threshold)
    ga.main(config, str(tmp_path / 'full'))
    ga.main({**config, 'timesteps': 64}, str(tmp_path / 'split'))
    first = read_state(tmp_path / 'split')
    assert first.it == 1
    ga.main(config, str(tmp_path / 'split'))
    full, split = (read_state(tmp_path / name) for name in ('full', 'split'))
    assert full.it == split.it == 2
    assert full.timesteps_so_far == split.timesteps_so_far == 128
    assert [o.seeds for o in full.population] == [o.seeds for o in split.population]
    assert [o.rewards for o in full.population] == [o.rewards for o in split.population]
    assert full.elite.seeds == split.elite.seeds
    for attr in ('rng_state', 'env_rng_state'):
        for a, b in zip(getattr(full, attr), getattr(split, attr)):
            np.testing.assert_array_equal(a, b)
    assert json.loads((tmp_path / 'full/best.json').read_text()) == json.loads(
        (tmp_path / 'split/best.json').read_text())
    assert len((tmp_path / 'split/progress.csv').read_text().splitlines()) == 3
    with pytest.raises(ValueError, match='Resume configuration changed: game'):
        ga.main({**config, 'game': 'pong'}, str(tmp_path / 'split'))


def test_best_uses_elite_score_not_candidate_average(tmp_path, monkeypatch):
    # First candidate pool averages 50, second averages 80; the first elite
    # still wins (100 versus 90). The previous code incorrectly replaced it.
    evaluator = AtariEvaluator('frostbite', 2, num_threads=1)
    validation_scores = iter(([100, 0], [70, 90]))

    def run(policy, tasks, max_steps=None):
        evaluator.steps_counter += len(tasks)
        return [(task.seeds, float(i), 1) for i, task in enumerate(tasks)]

    def repeated(policy, tasks, num_episodes, max_steps=None):
        scores = next(validation_scores) if len(tasks) > 1 else [123]
        return [(task.seeds, np.full(num_episodes, score), np.ones(num_episodes, dtype=int))
                for task, score in zip(tasks, scores)]

    monkeypatch.setattr(evaluator, 'run', run)
    monkeypatch.setattr(evaluator, 'run_repeated', repeated)
    monkeypatch.setattr(ga, 'AtariEvaluator', lambda *a, **k: evaluator)
    ga.main(tiny_config(timesteps=16), str(tmp_path))
    best = json.loads((tmp_path / 'best.json').read_text())
    assert best['validation_score'] == 100
    assert best['iteration'] == 1
    assert best['test_score'] == 123
    state = read_state(tmp_path)
    assert state.elite.seeds != state.curr_solution
    (tmp_path / 'best.json').write_text('{}')  # Simulate interrupted export.
    ga.load_state(str(tmp_path), tiny_config(timesteps=16))
    assert json.loads((tmp_path / 'best.json').read_text()) == best


@pytest.mark.parametrize('overrides', [
    {'population_size': 0}, {'selection_threshold': 5}, {'validation_threshold': 0},
    {'num_validation_episodes': 0}, {'num_test_episodes': 0}, {'test_max_steps': 0},
    {'timesteps': float('inf')}, {'num_envs': 0},
])
def test_invalid_config_fails_early(overrides):
    with pytest.raises(ValueError):
        ga.validate_config(tiny_config(**overrides))


@pytest.mark.parametrize('kwargs', [{'batch_size': 1}, {'grayscale': False},
                                    {'continuous': True}, {'frameskip': 2},
                                    {'autoreset_mode': 'NextStep'}, {'num_envs': 1}])
def test_unsupported_environment_modes(kwargs):
    with pytest.raises(ValueError):
        validate_env_kwargs(kwargs)


def test_replay_retains_action_space_and_sticky_actions(tmp_path):
    data = {'game': 'frostbite', 'model': 'Model', 'seeds': [1, [2, 0.002]],
            'env_kwargs': {'full_action_space': True, 'repeat_action_probability': 0.25}}
    path = tmp_path / 'best.json'
    path.write_text(json.dumps(data))
    game, model, seeds, kwargs = load_genome(path)
    assert (game, model, seeds) == ('frostbite', 'Model', (1, (2, 0.002)))
    assert kwargs == data['env_kwargs']


def test_ale_partial_reset_preserves_other_slots():
    kwargs = {'noop_max': 0, 'use_fire_reset': False}
    a = AtariEvaluator('frostbite', 2, num_threads=1, env_kwargs=kwargs)
    b = AtariEvaluator('frostbite', 2, num_threads=1, env_kwargs=kwargs)
    try:
        a.env.reset(seed=123)
        b.env.reset(seed=123)
        actions = np.array([1, 2])
        for _ in range(12):
            a.env.step(actions)
            obs, *_ = b.env.step(actions)
        obs = obs.copy()  # ALE may reuse observation buffers.
        reset_obs, _ = a.env.reset(options={'reset_mask': np.array([True, False])})
        np.testing.assert_array_equal(reset_obs[1], obs[1])
        for _ in range(8):
            a_out, b_out = a.env.step(actions), b.env.step(actions)
            for a_value, b_value in zip(a_out[:4], b_out[:4]):
                np.testing.assert_array_equal(a_value[1], b_value[1])
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA GPU not available'))])
@pytest.mark.parametrize('cutoff,frame_limit', [(3, 108000), (None, 16)])
def test_slot_refill_and_episode_limits(device, cutoff, frame_limit):
    evaluator = AtariEvaluator('frostbite', 2, num_threads=1, seed=0,
                               env_kwargs={'noop_max': 0, 'max_num_frames_per_episode': frame_limit})
    try:
        policy = BatchedPolicy('Model', evaluator.num_actions, 2, device=device)
        theta = torch.zeros(policy.num_params, device=device)
        tasks = [Task((i,), lambda: theta) for i in range(5)]
        results = evaluator.run(policy, tasks, max_steps=cutoff)
        assert [r[0] for r in results] == [t.seeds for t in tasks]
        expected_length = cutoff if cutoff is not None else frame_limit // 4
        assert [r[2] for r in results] == [expected_length] * len(tasks)
        assert evaluator.steps_counter == expected_length * len(tasks)
    finally:
        evaluator.close()


@pytest.mark.parametrize('mode', ['env_default', 'adaptive:2,0.5,2,8'])
def test_other_cutoff_modes(tmp_path, mode):
    config = tiny_config(episode_cutoff_mode=mode, timesteps=1,
                         env_kwargs={'noop_max': 0, 'max_num_frames_per_episode': 16})
    ga.main(config, str(tmp_path))
    state = read_state(tmp_path)
    assert state.it == 1
    assert state.tslimit == (None if mode == 'env_default' else 4)


def test_old_snapshot_rejected_with_replay_guidance(tmp_path):
    config = tiny_config()
    old = ga.TrainingState(config)
    del old.snapshot_version
    with open(tmp_path / 'snapshot.pkl', 'wb') as f:
        pickle.dump(old, f)
    with pytest.raises(ValueError, match='best.json can still be replayed'):
        ga.load_state(str(tmp_path), config)
