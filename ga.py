"""Deep GA for Atari (Such et al., 2017), ported from the Uber AI Labs / Chapter 10 code
to PyTorch + Gymnasium/ale-py.

    python ga.py -c configurations/ga_atari_config.json -o out/frostbite
"""

import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch

from neuroevolution import AtariEvaluator, BatchedPolicy, SharedNoiseTable, Task, make_schedule
from neuroevolution import logger as tlogger
from neuroevolution.evaluator import validate_env_kwargs
from neuroevolution.models import ARCHITECTURES

DEFAULTS = {
    "num_envs": 64,
    "num_threads": 0,
    "noise_table_size": 250_000_000,
    "device": "auto",
    "seed": None,
    "test_max_steps": None,
    "env_kwargs": {},
}

SNAPSHOT_VERSION = 1
# These can change without invalidating decoded genomes or the selection objective.
RESUME_OVERRIDES = {"device", "num_envs", "num_threads", "timesteps"}


def validate_config(config):
    for key in ("population_size", "validation_threshold", "num_validation_episodes",
                "num_test_episodes", "num_envs", "noise_table_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("selection_threshold", "num_threads"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("selection_threshold", "validation_threshold"):
        if config[key] > config["population_size"]:
            raise ValueError(f"{key} must not exceed population_size")
    if not math.isfinite(config["timesteps"]) or config["timesteps"] <= 0:
        raise ValueError("timesteps must be positive and finite")
    if config["model"] not in ARCHITECTURES:
        raise ValueError(f"Unknown model: {config['model']}")
    test_limit = config["test_max_steps"]
    if test_limit is not None and (type(test_limit) is not int or test_limit <= 0):
        raise ValueError("test_max_steps must be a positive integer or null")
    validate_env_kwargs(config["env_kwargs"])


class TrainingState:
    def __init__(self, config):
        self.snapshot_version = SNAPSHOT_VERSION
        self.config = dict(config)
        self.rng_state = None
        self.env_rng_state = None
        self.num_frames = 0
        self.population = []
        self.timesteps_so_far = 0
        self.time_elapsed = 0
        self.validation_timesteps_so_far = 0
        self.elite = None
        self.it = 0
        self.mutation_power = make_schedule(config["mutation_power"])
        self.curr_solution = None
        self.curr_solution_val = float("-inf")
        self.curr_solution_test = float("-inf")
        self.curr_solution_it = None

        mode = config["episode_cutoff_mode"]
        self.adaptive_tslimit = False
        if type(mode) is int and mode > 0:
            self.tslimit = mode
        elif isinstance(mode, str) and mode.startswith("adaptive:"):
            _, args = mode.split(":")
            arg0, arg1, arg2, arg3 = args.split(",")
            self.tslimit = int(arg0)
            self.incr_tslimit_threshold = float(arg1)
            self.tslimit_incr_ratio = float(arg2)
            self.tslimit_max = int(arg3)
            if not (0 < self.tslimit <= self.tslimit_max and
                    0 <= self.incr_tslimit_threshold <= 1 and self.tslimit_incr_ratio > 1):
                raise ValueError("Invalid adaptive episode cutoff settings")
            self.adaptive_tslimit = True
            tlogger.info(f"Starting timestep limit set to {self.tslimit}. When {self.incr_tslimit_threshold * 100}% "
                         f"of rollouts hit the limit, it will be increased by {self.tslimit_incr_ratio}")
        elif mode == "env_default":
            self.tslimit = None
        else:
            raise ValueError(f"Invalid episode_cutoff_mode: {mode!r}")

    def sample(self, schedule):
        return schedule.value(iteration=self.it, timesteps_so_far=self.timesteps_so_far)


class Offspring:
    def __init__(self, seeds, rewards, ep_len, validation_rewards=(), validation_ep_len=()):
        self.seeds = seeds
        self.rewards = rewards
        self.ep_len = ep_len
        self.validation_rewards = list(validation_rewards)
        self.validation_ep_len = list(validation_ep_len)

    @property
    def fitness(self):
        return np.mean(self.rewards)

    @property
    def training_steps(self):
        return np.sum(self.ep_len)


def resolve_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices are cpu, cuda, cuda:N, and auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Select a Colab GPU runtime, or use --device cpu.")
    return device


def save_best(log_dir, state):
    tmp = os.path.join(log_dir, "best.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"game": state.game, "model": state.model_name,
                   "env_kwargs": state.config["env_kwargs"],
                   "seeds": state.curr_solution, "validation_score": state.curr_solution_val,
                   "test_score": state.curr_solution_test, "iteration": state.curr_solution_it}, f)
    os.replace(tmp, os.path.join(log_dir, "best.json"))


def save_state(log_dir, state):
    tmp = os.path.join(log_dir, "snapshot.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, os.path.join(log_dir, "snapshot.pkl"))
    save_best(log_dir, state)


def load_state(log_dir, config):
    try:
        with open(os.path.join(log_dir, "snapshot.pkl"), "rb") as f:
            state = pickle.load(f)
    except FileNotFoundError:
        tlogger.info("No snapshot found, starting from scratch")
        return TrainingState(config)
    if getattr(state, "snapshot_version", None) != SNAPSHOT_VERSION:
        raise ValueError("This snapshot predates reliable resume support. Use a new output directory; "
                         "its best.json can still be replayed with play.py.")
    changed = [key for key in set(config) | set(state.config)
               if key not in RESUME_OVERRIDES and config.get(key) != state.config.get(key)]
    if changed:
        raise ValueError(f"Resume configuration changed: {', '.join(sorted(changed))}. Use a new output directory.")
    state.config = dict(config)
    # Repair the export if a disconnect occurred between the two file replacements.
    save_best(log_dir, state)
    tlogger.info(f"Loaded iteration {state.it} from {log_dir}/snapshot.pkl")
    return state


def main(config, out_dir):
    config = {**DEFAULTS, **config}
    validate_config(config)
    log_dir = out_dir or "out"
    tlogger.set_log_dir(log_dir)
    tlogger.info(json.dumps(config, indent=4, sort_keys=True))
    tlogger.info(f"Logging to: {log_dir}")

    device = resolve_device(config["device"])
    state = load_state(log_dir, config)
    state.game, state.model_name = config["game"], config["model"]
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    tlogger.info(f"Device: {device}")
    if device.type == "cuda":
        tlogger.info(f"GPU: {torch.cuda.get_device_name(device)}")
    all_tstart = time.time()

    evaluator = AtariEvaluator(config["game"], config["num_envs"], env_kwargs=config["env_kwargs"],
                               num_threads=config["num_threads"], seed=config["seed"])
    try:
        return train(config, log_dir, device, state, evaluator, all_tstart)
    finally:
        evaluator.close()


def train(config, log_dir, device, state, evaluator, all_tstart):
    policy = BatchedPolicy(config["model"], evaluator.num_actions, config["num_envs"], device=device,
                           obs_shape=evaluator.obs_shape)
    tlogger.info(policy.describe())
    noise = SharedNoiseTable(count=config["noise_table_size"], device=device)
    rs = np.random.RandomState(config["seed"])
    if state.rng_state is not None:
        rs.set_state(state.rng_state)
        evaluator.random.set_state(state.env_rng_state)
    if policy.num_params > len(noise):
        raise ValueError(f"noise_table_size must be at least {policy.num_params}")

    cached_parents = []
    selection_threshold = config["selection_threshold"]

    def select_parents():
        """Top ``selection_threshold`` genomes, always including the elite, with their weights cached."""
        top = state.population[:selection_threshold]
        parents = top if state.elite in top else [state.elite] + top[:selection_threshold - 1]
        return [(policy.compute_weights_from_seeds(noise, o.seeds, cache=cached_parents), o.seeds) for o in parents]

    if state.population and selection_threshold > 0:
        tlogger.info("Caching parents")
        cached_parents.extend(select_parents())

    def make_offspring():
        if not cached_parents:
            seeds = policy.randomize(rs, noise)
            return Task(seeds, lambda: policy.compute_weights_from_seeds(noise, seeds))
        parent_theta, parent_seeds = cached_parents[rs.randint(len(cached_parents))]
        idx = noise.sample_index(rs, policy.num_params)
        power = state.sample(state.mutation_power)
        return Task(parent_seeds + ((idx, power),),
                    lambda: policy.compute_mutation(noise, parent_theta, idx, power))

    def task_from_seeds(seeds):
        return Task(seeds, lambda: policy.compute_weights_from_seeds(noise, seeds, cache=cached_parents))

    tstart = time.time()
    starting_timesteps = state.timesteps_so_far
    while state.timesteps_so_far < config["timesteps"]:
        tstart_iteration = time.time()
        frames_before = evaluator.steps_counter
        iteration_mutation_power = state.sample(state.mutation_power)

        tasks = [make_offspring() for _ in range(config["population_size"])]
        results = [Offspring(seeds, [rew], [length])
                   for seeds, rew, length in evaluator.run(policy, tasks, max_steps=state.tslimit)]
        state.num_frames += evaluator.steps_counter - frames_before

        state.it += 1
        tlogger.record_tabular("Iteration", state.it)
        tlogger.record_tabular("MutationPower", iteration_mutation_power)

        rewards = np.array([a.fitness for a in results])
        population_timesteps = sum(a.training_steps for a in results)
        state.population = sorted(results, key=lambda x: x.fitness, reverse=True)
        tlogger.record_tabular("PopulationEpRewMax", np.max(rewards))
        tlogger.record_tabular("PopulationEpRewMean", np.mean(rewards))
        tlogger.record_tabular("PopulationEpCount", len(rewards))
        tlogger.record_tabular("PopulationTimesteps", population_timesteps)
        tlogger.record_tabular("NumSelectedIndividuals", selection_threshold)

        tlogger.info("Evaluate population")
        validation_population = state.population[:config["validation_threshold"]]
        if state.elite is not None:
            validation_population = [state.elite] + validation_population[:-1]

        validation = evaluator.run_repeated(policy, [task_from_seeds(o.seeds) for o in validation_population],
                                            num_episodes=config["num_validation_episodes"],
                                            max_steps=state.tslimit)
        population_validation = [np.mean(rews) for _, rews, _ in validation]
        population_validation_len = [np.sum(lens) for _, _, lens in validation]

        population_elite_idx = int(np.argmax(population_validation))
        state.elite = validation_population[population_elite_idx]
        _, population_elite_evals, population_elite_evals_timesteps = evaluator.run_repeated(
            policy, [task_from_seeds(state.elite.seeds)], num_episodes=config["num_test_episodes"],
            max_steps=config["test_max_steps"])[0]

        validation_timesteps = sum(population_validation_len)
        timesteps_this_iter = population_timesteps + validation_timesteps
        state.timesteps_so_far += timesteps_this_iter
        state.validation_timesteps_so_far += validation_timesteps

        tlogger.record_tabular("TruncatedPopulationRewMean", np.mean([a.fitness for a in validation_population]))
        tlogger.record_tabular("TruncatedPopulationValidationRewMean", np.mean(population_validation))
        tlogger.record_tabular("TruncatedPopulationEliteValidationRew", np.max(population_validation))
        tlogger.record_tabular("TruncatedPopulationEliteIndex", population_elite_idx)
        tlogger.record_tabular("TruncatedPopulationEliteTestRewMean", np.mean(population_elite_evals))
        tlogger.record_tabular("TruncatedPopulationEliteTestEpCount", len(population_elite_evals))
        tlogger.record_tabular("TruncatedPopulationEliteTestEpLenSum", np.sum(population_elite_evals_timesteps))

        elite_validation = float(population_validation[population_elite_idx])
        if elite_validation > state.curr_solution_val:
            state.curr_solution = state.elite.seeds
            state.curr_solution_val = elite_validation
            state.curr_solution_test = float(np.mean(population_elite_evals))
            state.curr_solution_it = state.it

        elapsed = time.time() - tstart_iteration
        state.time_elapsed += elapsed
        tlogger.record_tabular("ValidationTimestepsThisIter", validation_timesteps)
        tlogger.record_tabular("ValidationTimestepsSoFar", state.validation_timesteps_so_far)
        tlogger.record_tabular("TimestepsThisIter", timesteps_this_iter)
        tlogger.record_tabular("TimestepsPerSecondThisIter", timesteps_this_iter / elapsed)
        tlogger.record_tabular("TimestepsComputed", state.num_frames)
        tlogger.record_tabular("TimestepsSoFar", state.timesteps_so_far)
        tlogger.record_tabular("TimeElapsedThisIter", elapsed)
        tlogger.record_tabular("TimeElapsedThisIterTotal", elapsed)
        tlogger.record_tabular("TimeElapsed", state.time_elapsed)
        tlogger.record_tabular("TimeElapsedTotal", time.time() - all_tstart)
        tlogger.dump_tabular()

        tlogger.info(f"Current elite: {len(state.elite.seeds) - 1} mutations deep")
        fps = (state.timesteps_so_far - starting_timesteps) / (time.time() - tstart)
        tlogger.info(f"Timesteps per second: {fps:.0f}. Elapsed: {(time.time() - all_tstart) / 3600:.2f}h "
                     f"ETA {max(0, config['timesteps'] - state.timesteps_so_far) / fps / 3600:.2f}h")

        if state.adaptive_tslimit:
            if np.mean([a.training_steps >= state.tslimit for a in results]) > state.incr_tslimit_threshold:
                state.tslimit = int(min(state.tslimit * state.tslimit_incr_ratio, state.tslimit_max))
                tlogger.info(f"Increased threshold to {state.tslimit}")

        state.rng_state = rs.get_state()
        state.env_rng_state = evaluator.random.get_state()
        save_state(log_dir, state)
        tlogger.info(f"Saved iteration {state.it} to {log_dir}/snapshot.pkl")

        if selection_threshold > 0:
            new_parents = select_parents()
            cached_parents.clear()
            cached_parents.extend(new_parents)

    tlogger.info(f"Training terminated after {state.timesteps_so_far} timesteps")
    return float(state.curr_solution_test), float(state.curr_solution_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-accelerated Deep GA for Atari (PyTorch + ale-py).")
    parser.add_argument("-c", "--config", type=str, required=True, help="The experiment configuration to use.")
    parser.add_argument("-o", "--out", type=str, default=None, help="The output directory to store results.")
    parser.add_argument("--device", help="Override config device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-envs", type=int, help="Override parallel slot count")
    parser.add_argument("--timesteps", type=float, help="Override total budget, including completed steps on resume")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    for key in ("device", "num_envs", "timesteps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    test_score, eval_score = main(config, out_dir=args.out)
    print(f"Test score: {test_score:f}, evaluation score: {eval_score:f}")
