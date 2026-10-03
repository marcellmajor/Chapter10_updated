# Chapter 10: Deep GA for Atari with PyTorch and ALE

This self-contained port replaces Chapter 10's TensorFlow 1.x and custom C++ TensorFlow
op with PyTorch and `ale_py.vector_env.AtariVectorEnv`. It implements the genetic
algorithm (GA) and random-search baseline. Evolution strategies (`es.py`), the maze,
novelty search, and the original snapshot visualization tools are not ported.

**CPU training, checkpoint resume, and replay are tested. Colab/CUDA execution has not
been verified on hardware here.** The commands below explicitly check CUDA and exercise
the GPU paths before a long run. The Colab configuration is a conservative starting
point, not a measured optimum or a promise to finish in one session.

## What the example does

Chapter 10, "Deep Neuroevolution", of *Hands-On Neuroevolution with Python* (Iaroslav
Omelianenko, Packt, 2019) trains an agent to play the Atari 2600 game **Frostbite** from
screen pixels alone. The controller is a convolutional network with about 4 million
parameters, found by a genetic algorithm instead of backpropagation. The method is the
deep GA of Such et al., [Deep Neuroevolution: Genetic Algorithms Are a Competitive
Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567)
(Uber AI Labs, 2017). The chapter's code builds on Uber's
[deep-neuroevolution](https://github.com/uber-research/deep-neuroevolution) repository.

Gradient-based deep reinforcement learning methods such as DQN and A3C train a network to
predict values or action probabilities and update it from gradients of a loss. The deep GA
computes no gradients. It plays the game with many networks, keeps the best scorers, and
builds the next generation from randomly perturbed copies of them. The only learning
signal is the game score at the end of each episode.

### The game

In Frostbite, the player jumps between four rows of ice floes drifting over water. Landing
on a white floe adds a block to an igloo on the shore and changes the floe's color. The
player must finish the igloo before a 45-second timer freezes them, then enter it to
clear the level. Faster levels earn larger bonuses, and hazards must be avoided along the
way. The agent is told none of these rules: it sees only screen frames, and its reward is
the game score.

### From screen to action

The [Arcade Learning Environment (ALE)](https://ale.farama.org/) emulates the console. For
each agent step, it repeats the chosen action for 4 frames, max-pools the last two frames
to remove sprite flicker, converts to grayscale, and resizes to 84×84. The four most recent
processed frames form the network input. Each episode starts with up to 30 random no-op
actions, so a policy cannot simply memorize one button sequence for an otherwise
deterministic game.

The policy is the DQN-style `LargeModel` described in the book, with TensorFlow `SAME`
padding:

| Layer | Configuration | Output shape |
|---|---|---|
| Input | 4 stacked grayscale frames, scaled to [0, 1] | 4×84×84 |
| Conv 1 | 32 filters, 8×8, stride 4, ReLU | 32×21×21 |
| Conv 2 | 64 filters, 4×4, stride 2, ReLU | 64×11×11 |
| Conv 3 | 64 filters, 3×3, stride 1, ReLU | 64×11×11 |
| Dense | 512 units, ReLU | 512 |
| Output | One score per action | 18 for Frostbite |

The agent plays the action with the highest score. In total, `LargeModel` has 4,052,658
weights and biases. The smaller `Model` has 1,008,450 parameters: two convolutional
layers with 16 and 32 filters and a 256-unit dense layer. It is used by the smoke and
random-search configurations. The book calls the network a Q-value approximator because
the architecture comes from DQN. However, the GA never trains the outputs to predict
returns. Only their argmax matters, and selection sees only episode scores.

### A genome is a list of seeds

Storing every weight would take 16 MB per `LargeModel` individual, so a population of
1,000 would be expensive to keep and pass around. With Uber's encoding, a genome is
instead the list of seeds that produced it:

```text
genome = (τ0, (τ1, σ), (τ2, σ), ..., (τn, σ))
θ0     = φ(τ0)                  initialization
θk     = θk-1 + σ · ε(τk)       mutation k, for k = 1..n
```

A seed τ is not given to a random number generator at runtime. It is an offset into a
fixed table of 250 million Gaussian numbers, generated once from `RandomState(123)`.
ε(τ) is the slice starting at τ, with one value per parameter. Looking up a slice is much
faster than sampling millions of new random numbers for every mutation. During
initialization, φ scales each layer's slice by `std / sqrt(fan_in)` and sets biases to zero.

A genome gains one entry per generation of ancestry, regardless of network size. The
supplied `pretrained/frostbite_uber_elite.json` has 266 entries: 1 initialization and 265
mutations at σ = 0.002. It decodes into all 4,052,658 parameters. The book's sample run
printed an elite 26 mutations deep, and `ga.py` logs the current elite's depth each
generation. Decoding does not replay full chains during training. The parents' weights
are cached, so building a child costs one table slice and one vector addition.

### One generation of the GA

The steps use `ga_atari_config.json`, the book's configuration. The
[smaller presets](#configuration-and-resource-sizing) scale these counts down.

1. **Create offspring.** Build `population_size` (1,000) genomes. In the first generation,
   each is a fresh random initialization. Later, each one picks one of the
   `selection_threshold` (20) parents uniformly at random and adds one Gaussian mutation
   with `mutation_power` σ = 0.002. There is no crossover. The network topology is fixed;
   only the weights evolve.
2. **Evaluate.** Each genome plays one episode, and the score is its fitness. Training
   episodes stop after `episode_cutoff_mode`: 5,000 agent steps, or the 20,000 frames
   given in the book. Up to `num_envs` games run at once, each slot with its own network.
3. **Validate.** One episode is a noisy estimate. The `validation_threshold` (10) best
   scorers therefore play `num_validation_episodes` (30) more episodes each. From the
   second generation, the current elite replaces the tenth candidate and must keep its
   place. The candidate with the highest validation mean becomes the new elite.
4. **Test.** The elite plays `num_test_episodes` (200) episodes without the training
   cutoff; ALE's 108,000-frame episode limit still applies. The test mean only reports
   progress. It does not affect selection and does not count toward `timesteps`.
5. **Save.** The snapshot is written. `best.json` is updated when the elite's validation
   mean is the highest seen so far.
6. **Select parents.** The 20 genomes with the best training scores become parents. The
   elite always joins them, replacing the twentieth if needed. Their weights are cached.
   Training continues until the `timesteps` budget of training plus validation steps
   (1.5 billion) is reached.

The book describes elitism as copying the elite unchanged into the next generation. In
the code, the elite instead survives through validation and parent selection. Every
training genome in a generation is a new mutant. Each generation of this configuration
plays up to 1,500 episodes: 1,000 training, 300 validation, and 200 test.

### Random-search baseline

`rs_atari_config.json` sets `selection_threshold` to 0. With no parents, every genome in
every generation is a new random network with a one-seed genome. Validation, elite
selection, and testing are unchanged. The run reports the best network found by random
sampling, which shows how much the GA's mutation and selection add. Such et al. use
this baseline; the book chapter covers only the GA. This configuration uses the smaller
`Model`; set `"model": "LargeModel"` to match the GA configuration.

### Reading the training output

Each generation prints a table and appends it to `progress.csv`. The main score columns
are:

| Column | Meaning |
|---|---|
| `PopulationEpRewMax`, `PopulationEpRewMean` | Best and mean single-episode training scores in the population |
| `TruncatedPopulationRewMean` | Mean training score of the validation candidates |
| `TruncatedPopulationValidationRewMean` | Mean validation score across the candidates |
| `TruncatedPopulationEliteValidationRew` | New elite's validation mean, used to select it |
| `TruncatedPopulationEliteIndex` | Elite's position among candidates; after generation 1, 0 means the previous elite was kept |
| `TruncatedPopulationEliteTestRewMean` | Elite's mean test score, the best estimate of actual performance |
| `MutationPower` | σ used for this generation's mutations |

`PopulationEpRewMax` is the best of many noisy single episodes, so it is optimistic.
Validation and test scores are usually lower. In the book's sample run, the population
maximum was 3,470, the elite's validation mean 3,100, and its 200-episode test mean
3,060. This port's preprocessing differs from the original
([details](#fidelity-and-validation)), so its scores need not match. The book's text
describes some `TruncatedPopulation*` columns differently; the definitions above follow
the code.

### From the book's code to this port

| Original Chapter 10 code | This port |
|---|---|
| `gym_tensorflow` custom TensorFlow op with an `atari-py` fork (`AtariEnv`) | `ale_py.vector_env.AtariVectorEnv` in [`neuroevolution/evaluator.py`](neuroevolution/evaluator.py) |
| `RLEvalutionWorker` and `ConcurrentWorkers` | `AtariEvaluator.run` and `run_repeated`: a slot scheduler with partial resets |
| `models/dqn.py` (`Model`, `LargeModel`) | `ARCHITECTURES` and `BatchedPolicy` in [`neuroevolution/models.py`](neuroevolution/models.py) |
| `models/base.py` (`compute_weights_from_seeds`, `compute_mutation`, `mutate`) | `BatchedPolicy` methods, plus `make_offspring` in [`ga.py`](ga.py) |
| `SharedNoiseTable` | [`neuroevolution/noise.py`](neuroevolution/noise.py), stored on the training device |
| `ga.py` experiment runner | [`ga.py`](ga.py), with the same configuration keys plus resume |
| `display.py`, which opens a game window | [`play.py`](play.py): headless replay with MP4 recording |
| VINE, plus the `master_extract_parent_ga` and `master_extract_cloud_ga` helpers | Not ported; plot `progress.csv` instead |

The chapter's first exercise, increasing `population_size`, works unchanged. Use a new
output directory, because resume rejects configuration changes.

## Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcellmajor/Chapter10_updated/blob/main/colab/deep_ga_atari.ipynb)

The code is published at
[github.com/marcellmajor/Chapter10_updated](https://github.com/marcellmajor/Chapter10_updated).
Open [`colab/deep_ga_atari.ipynb`](colab/deep_ga_atari.ipynb) with the badge above,
select a GPU runtime, and run the cells individually, in order: setup, GPU diagnostics,
tests, and smoke training. The repository is public, so no GitHub sign-in or token is
needed.

The setup cell clones the `main` branch into `/content/Chapter10_updated` on the VM's
local disk. Rerunning it on the same VM fast-forwards that checkout to the latest pushed
commit. Results are kept in `MyDrive/chapter10-runs/<RUN_NAME>/`. The notebook starts
real training with a 1-million-step budget; increase `TOTAL_STEPS` only when ready to continue. Diagnostics include package
versions, GPU/CPU details, the Git revision, and logs for each command. Share
`diagnostics/environment.json`, the test summary, and any failing command's output
when troubleshooting. The notebook structure and Python cells were checked locally;
its Google Drive and CUDA workflow still needs execution on a real Colab runtime.

The cells below are an alternative manual setup with the source itself stored in Drive.

## Google Colab setup

1. In **Runtime → Change runtime type**, select an available GPU. Colab's GPU models,
   CPU resources, and session limits [vary with availability](https://research.google.com/colaboratory/faq.html).
   High-RAM increases **host** memory, not GPU memory.
2. Run these notebook cells in order. The first cell clones the repository into
   `MyDrive/Chapter10_updated/` if that folder does not exist yet. Mounting Drive keeps
   training outputs across VM replacement; a checkpoint is written after each completed
   generation. To get later changes, run
   `!git -C /content/drive/MyDrive/Chapter10_updated pull --ff-only`. The original book
   repository does not contain this port.

```python
from google.colab import drive
drive.mount('/content/drive')
!test -d /content/drive/MyDrive/Chapter10_updated || git clone https://github.com/marcellmajor/Chapter10_updated.git /content/drive/MyDrive/Chapter10_updated
%cd /content/drive/MyDrive/Chapter10_updated
```

```python
# No --upgrade: retain Colab's CUDA-enabled torch when it meets requirements.txt.
%pip install -q -r requirements.txt
```

If Colab asks for a runtime restart after installation, restart, remount Drive, and
repeat the working-directory cell before continuing. Python 3.10+ is required.
`ale-py>=0.12.1,<0.13` is intentional: 0.12.1 fixes partial resets used by the
scheduler, and 0.12.0 omitted ROMs from its wheels. No AutoROM installation, ROM
import, TensorFlow build, or custom CUDA extension is needed with the supported ALE
wheels. See the [ALE release notes](https://ale.farama.org/release_notes/).

```python
import os, sys
import torch, gymnasium, ale_py, numpy
print(sys.version)
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('gymnasium:', gymnasium.__version__, 'ale-py:', ale_py.__version__, 'numpy:', numpy.__version__)
print('CPU cores:', os.cpu_count())
assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
print('GPU:', torch.cuda.get_device_name(0))
!nvidia-smi
```

```python
# Tests include CPU and CUDA network comparisons and actual ALE slot scheduling.
# CUDA tests skip on a CPU machine; the assertion above prevents a silent CPU-only check.
!python -m pytest -q tests
!python ga.py -c configurations/ga_smoke_config.json -o out/smoke-v1 --device cuda
```

The smoke configuration uses the smaller `Model`, 8 slots, short episodes, and a
12,000-agent-step budget. It checks execution, not learning quality; elapsed time and
number of generations depend on the runtime and episode lengths. To actually test
resume after the budget has been reached, increase the **total** budget:

```python
!python ga.py -c configurations/ga_smoke_config.json -o out/smoke-v1 --device cuda --timesteps 24000
```

Optionally replay the supplied original Frostbite policy before training. This exercises
`LargeModel` and reconstructs a nearly 1 GB noise-table prefix. Scores depend on
preprocessing, environment starts, and numerical precision; they are not a pass/fail test.

```python
!python play.py --seeds pretrained/frostbite_uber_elite.json --episodes 3 --device cuda --no-video
```

Start the Colab experiment with a short budget first. It uses `LargeModel`, population
256, 64 simultaneous slots, 10 validation candidates × 10 episodes, and 32 elite test
episodes per generation. The configuration requests CUDA and fails clearly if it is
unavailable.

```python
!python ga.py -c configurations/ga_colab_config.json -o out/frostbite-v1 --timesteps 1000000
```

The budget is checked **between generations**, so the last generation can overshoot it.
Once throughput and memory look reasonable, resume toward the configuration's full
50-million-step budget:

```python
!python ga.py -c configurations/ga_colab_config.json -o out/frostbite-v1
```

```python
!python play.py --run out/frostbite-v1 --episodes 5 --device cuda
from IPython.display import Video
Video('out/frostbite-v1/best_frostbite.mp4', embed=True)
```

Replay records the first episode as a grayscale MP4 of the policy's input, enlarged to
336×336 at 15 fps, and reports all episode scores. Frames stream to ffmpeg to keep host
memory bounded. Use `--video /path/movie.mp4` to change the destination, `--no-video`
for scores only, or `--max-steps 1000` for a short preview. Training itself does not
record videos.

## Resume and saved results

Use the same configuration and output directory to resume the last completed generation.
An interrupted generation is repeated. Re-running a completed budget does no further
training; increase `--timesteps` to continue. Do not run two training processes against
the same output directory.

| File | Contents |
|---|---|
| `config.json` | Effective configuration, including CLI overrides |
| `snapshot.pkl` | Population, elite, mutation schedule, counters, best policy, configuration, and genome/environment RNG states |
| `best.json` | Highest observed elite validation mean, its genome, test mean, discovery iteration, and environment settings |
| `progress.csv` | One metric row per completed evaluation generation |
| `log.txt` | Configuration and progress messages |

Snapshots and `best.json` are replaced via temporary files. `best.json` is repaired from
the snapshot on resume if export was interrupted. CSV/log entries are diagnostic and
can contain a repeated iteration if a disconnect occurs after logging but before saving.
Use only trusted `snapshot.pkl` files: pickle loads Python objects.

You may change `timesteps`, `device`, `num_envs`, and `num_threads` on resume. Other
configuration changes are rejected to avoid mixing experiments. RNG state is restored,
but changing slot count, hardware, or library versions may change results; bit-identical
CUDA trajectories are not guaranteed. The `seed` option seeds both genome sampling and
per-phase environment resets. `null` selects fresh random streams at the start of a run.

Snapshots from the initial version of `Chapter10_updated` lack configuration and RNG
state and are rejected with an explanation. Start these revised experiments in a **new
output directory**; existing `best.json` files still work with `play.py`. Old exports
without `env_kwargs` use the default environment settings.

## Configuration and resource sizing

CLI overrides: `--device auto|cpu|cuda|cuda:N`, `--num-envs N`, and `--timesteps N`.
`auto` selects CUDA when available and otherwise CPU; the Colab config explicitly uses
`cuda`. CPU is suitable for smoke tests. Apple MPS is not a supported device for this port.

| Key | Meaning |
|---|---|
| `game` | ALE ROM name such as `frostbite`, `pong`, or `space_invaders` |
| `model` | `Model` (1,008,450 parameters with 18 actions) or `LargeModel` (4,052,658) |
| `population_size` | Genomes per generation, one training episode each |
| `selection_threshold` | Number of parents; 0 selects the random-search baseline |
| `validation_threshold` | Candidates re-evaluated to select the elite, including the previous elite after generation 1 |
| `num_validation_episodes` | Episodes per validation candidate |
| `num_test_episodes` | Test episodes for the selected elite every generation |
| `mutation_power` | Gaussian mutation standard deviation: a number or a schedule object |
| `episode_cutoff_mode` | Positive agent-step limit, `"env_default"`, or `"adaptive:start,threshold,ratio,max"` |
| `timesteps` | Total training **plus validation** agent-step budget; excludes elite test episodes |
| `num_envs` | Simultaneous environment slots and policy batch size (default 64) |
| `num_threads` | ALE emulator threads, 0 = automatic |
| `noise_table_size` | Number of float32 values in the fixed noise table (default 250,000,000) |
| `device` | `auto` by default; Colab config sets `cuda` |
| `seed` | Integer random seed or `null` (default) |
| `test_max_steps` | Optional test episode step limit; `null` uses ALE's episode limit |
| `env_kwargs` | ALE options, e.g. `{"repeat_action_probability": 0.25}`; saved for replay |

Population, validation, and test counts must be positive; selection may be zero.
Selection and validation thresholds cannot exceed the population. The noise table must
fit at least one flat policy vector. Scheduling/input options are fixed: synchronous
batching, same-step autoreset, discrete actions, grayscale 84×84 observations, stack 4,
and frameskip 4. Unsupported `env_kwargs` overrides fail early. Options such as sticky
actions, full action space, fire reset, or reward clipping change the experiment;
`play.py --run` retains them.

Mutation schedules accept `ConstantSchedule` (`value`), `LinearSchedule`, or
`ExponentialSchedule` (`initial_p`, `final_p`, `schedule`, `field`). For example:

```json
{"type": "LinearSchedule", "initial_p": 0.01, "final_p": 0.002,
 "schedule": 100, "field": "iteration"}
```

The schedule field can be `iteration` or `timesteps_so_far`. Adaptive cutoffs increase
when the fraction of training episodes hitting the limit exceeds `threshold`; the
increase is by `ratio`, capped at `max`. Comparing best historical validation scores
across changing cutoffs is imperfect because the evaluation horizon changes.

One agent step advances up to 4 emulator frames (fewer if an episode ends during the
skip). `TimestepsSoFar` counts training plus validation; `TimestepsComputed` counts only
active training steps. `TimestepsPerSecondThisIter` divides budgeted steps by total
generation time, including test evaluation, so it is useful for estimating budget ETA.
Idle slots and reset no-ops add work that these counters do not count.

| Configuration | Intended use |
|---|---|
| `ga_smoke_config.json` | Small CPU/GPU execution check; 12,000 steps |
| `ga_colab_config.json` | Initial CUDA experiment; 50 million steps, adjustable by CLI |
| `ga_atari_config.json` | Original chapter's larger GA settings; 1.5 billion steps |
| `rs_atari_config.json` | Original chapter's random-search settings; 1.5 billion steps |

The original JSON budget is in **agent steps, not emulator frames**: 1.5 billion steps
corresponds to roughly 6 billion frames, before test episodes and resets. Neither large
configuration is a short Colab example. Estimate duration from measured throughput;
50 million budgeted steps at 1,000 steps/s is about 13.9 hours.

The 250-million-entry noise table uses 1.00 GB (0.93 GiB) on the selected device and
roughly 1.1 GB of host RAM during construction. `LargeModel` slot weights use about
16.2 MB each: approximately 1.04 GB at 64 slots or 4.15 GB at 256 slots. Add parent
caches (20 parents ≈ 324 MB), activations, temporary weights, CUDA/cuDNN workspaces,
allocator overhead, and emulator memory. These figures are not total peak memory.

The [ALE vector environment](https://ale.farama.org/vector-environment/) runs emulators
on CPU; this port runs policy inference and genome reconstruction on the chosen device.
It uses grouped convolutions and batched matrix multiplies with distinct weights for
each slot. All slots still execute during phase tails, including already-finished slots.
More GPU memory alone does not guarantee higher throughput. If CUDA runs out of memory,
retry with `--num-envs 32` or `16`; reducing slots does not reduce the population size.
Benchmark 32/64/128 slots on the actual VM before scaling further. Tuning ideas are in
[`PLAN.md`](PLAN.md).

## Local setup

```bash
git clone https://github.com/marcellmajor/Chapter10_updated.git
cd Chapter10_updated
python3 -m venv .venv
source .venv/bin/activate
# For CUDA, install the appropriate torch wheel first using the PyTorch selector.
pip install -r requirements.txt
python -m pytest -q tests
python ga.py -c configurations/ga_smoke_config.json -o out/smoke-v1 --device cpu
```

Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for a local CUDA installation. The local CPU verification environment is macOS arm64,
Python 3.12.2, torch 2.14.1, gymnasium 1.3.0, ale-py 0.12.1, and NumPy 2.5.3.
Minimum dependency versions are compatibility requirements, not a fully tested matrix.

## Fidelity and validation

The genome encoding remains `(init_index, (mutation_index, power), ...)`. The noise table
uses the original `RandomState(123).randn` stream, float32 weights, TensorFlow `SAME`
padding, and the original flat parameter order (including NHWC flattening). Original
seed lists therefore reconstruct the same parameter layout and noise values. The
included `pretrained/frostbite_uber_elite.json` comes from `Chapter10/display.py`.

ALE supplies grayscale conversion, resize, max-pooling, 4-frame stacking, and random
no-op starts. Defaults disable sticky actions, fire reset, episodic life, and reward
clipping; the default episode limit is 108,000 emulator frames. Resize/grayscale
implementation and reset details differ from the old custom op, so trajectories and
scores need not match the original. Natural episode ends use same-step autoreset;
artificial training cutoffs reset only the affected slots.

Regression tests cover noise equality, both network architectures against a CPU
reference, cached mutation reconstruction, actual ALE partial resets, slot refilling,
cutoffs and environment termination, best-policy selection, configuration checks, and
uninterrupted versus resumed runs for both GA and random search. CUDA variants run
when a GPU is available and otherwise show as skipped. A successful smoke run verifies
the pipeline; it does not establish Atari learning performance or reproduce paper results.

Review verification (2026-10-03, CPU environment above): **28 tests passed, 3 CUDA tests
skipped**. The smoke CLI reached 14,400
budgeted steps in 3 generations (overshooting its 12,000-step target); resume toward
24,000 steps continued that run. The supplied Frostbite elite scored 2,800, 2,610, and
2,860 with `play.py --episodes 3 --seed 0 --device cpu --no-video` (mean 2,756.7).
CUDA numerical tests disable TF32 to compare full float32 arithmetic; training uses
PyTorch's default precision settings. No GPU timing or learning benchmark is claimed.
