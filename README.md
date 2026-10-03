# Chapter 10: Deep GA for Atari with PyTorch and ALE

This self-contained port replaces Chapter 10's TensorFlow 1.x and custom C++ TensorFlow
op with PyTorch and `ale_py.vector_env.AtariVectorEnv`. It implements the genetic
algorithm (GA) and random-search baseline. Evolution strategies (`es.py`), the maze,
novelty search, and the original snapshot visualization tools are not ported.

**CPU training, checkpoint resume, and replay are tested. Colab/CUDA execution has not
been verified on hardware here.** The commands below explicitly check CUDA and exercise
the GPU paths before a long run. The Colab configuration is a conservative starting
point, not a measured optimum or a promise to finish in one session.

## Standalone repository and Colab notebook

The ready-to-run notebook is [`colab/deep_ga_atari.ipynb`](colab/deep_ga_atari.ipynb).
Copy the **contents** of this directory into your new repository directory, including
`.gitignore` and `LICENSE`; omit `.venv/`, `out/`, `__pycache__/`, and `.pytest_cache/`.
The included license preserves the original source notices. `ga.py`, `requirements.txt`,
and this README should be at the new repository root.

Create an empty repository on GitHub, then run these commands in your new local
directory, replacing the remote URL with yours:

```bash
git init -b main
git add .
git status --short
git commit -m "Add modern Chapter 10 Atari GA example"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Open Google Colab, choose **File → Open notebook → GitHub**, paste your repository
URL, and select `colab/deep_ga_atari.ipynb`. Alternatively, upload that notebook file
directly to Colab. Select a GPU runtime and set `REPO_URL` in the first code cell.
Run cells individually, in order, starting with setup, GPU diagnostics, tests, and smoke
training. The notebook supports code either at the repository root or under
`Chapter10_updated/`; it also has a manually uploaded source option for private repos.

The notebook clones code onto the VM's local disk and keeps results in
`MyDrive/chapter10-runs/<RUN_NAME>/`. It starts real training with a 1-million-step
budget; increase `TOTAL_STEPS` only when ready to continue. Diagnostics include package
versions, GPU/CPU details, the Git revision, and logs for each command. Share
`diagnostics/environment.json`, the test summary, and any failing command's output
when troubleshooting. The notebook structure and Python cells were checked locally;
its Google Drive and CUDA workflow still needs execution on a real Colab runtime.

The cells below are an alternative manual setup with the source itself stored in Drive.

## Google Colab setup

1. In **Runtime → Change runtime type**, select an available GPU. Colab's GPU models,
   CPU resources, and session limits [vary with availability](https://research.google.com/colaboratory/faq.html).
   High-RAM increases **host** memory, not GPU memory.
2. Copy `Chapter10_updated/` into `MyDrive/Chapter10_updated/` (omit `.venv/`, caches,
   and old outputs). Alternatively, clone a repository that contains this updated directory
   and adjust the working-directory cell below. The original book repository may not
   contain these updates.
3. Run these notebook cells in order. Mounting Drive keeps training outputs across VM
   replacement; a checkpoint is written after each completed generation.

```python
from google.colab import drive
drive.mount('/content/drive')
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
