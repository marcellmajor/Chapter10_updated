# Plan: from initial port to a tuned, working Colab GPU example

## Phase 0: initial port (done)

- [x] Replace the TF1 + `gym_tensorflow.so` stack with PyTorch + `ale_py.vector_env.AtariVectorEnv`.
- [x] `BatchedPolicy`: one weight set per environment slot, evaluated with grouped
      convolutions and batched matmuls. Uses TF "SAME" padding and the original flat
      parameter layout.
- [x] `SharedNoiseTable` generated exactly like the original (bit-identical), held on the GPU.
- [x] `AtariEvaluator`: slot scheduler with lazy weight construction, same-step autoreset,
      explicit resets for step-limited episodes.
- [x] `ga.py` port with the original config keys, parent caching, elite selection,
      validation/test episodes, adaptive step limit, snapshot/resume and `best.json`.
- [x] `play.py` (headless MP4 replay) and the original pretrained Frostbite elite as JSON.
- [x] Tests: batched net vs reference net, noise-table equality, parameter counts.
- [x] CPU verification: smoke run and resume work; pretrained elite scores about 2,760.

## Phase 1: validate on a Colab GPU (next session)

The CPU review and regression checks are complete; see the README for the supported
setup and resume restrictions. ALE now requires 0.12.1 for correct partial resets.
GPU execution and throughput below remain unverified.

1. Run the README cells on an A100/H100 runtime: `pytest`, pretrained replay, smoke config
   with `"device": "cuda"` (the `auto` default already chooses CUDA).
2. Measure throughput (`TimestepsPerSecondThisIter`) for `num_envs` in {32, 64, 128, 256}
   and `num_threads` in {0, `nproc`}. Record GPU utilization (`nvidia-smi dmon`) and
   CPU utilization. Measure whether emulation, grouped inference, or transfers dominate.
3. Record memory: noise table (1 GB) + `num_envs` × 16 MB for `LargeModel` weights.
4. Tune `ga_colab_config.json` from these measurements (it currently uses a conservative,
   unbenchmarked 64 slots), and estimate how long a 5e7-step run takes including tests.

## Phase 2: performance work (depends on phase 1 numbers)

- **Overlap CPU and GPU**: use ale-py's async mode (`batch_size < num_envs`, `send`/`recv`)
  so that the emulator steps one half of the slots while the GPU runs the other half. The
  original code overlapped the two in the same way.
- **Tail waste**: at the end of each phase, finished slots keep stepping idle. Shrink the
  active batch (index only running slots) or start the next phase's tasks early.
- **Transfers**: pinned host buffers and `non_blocking=True` copies for observations and actions.
- **Inference speed**: try `torch.compile`, bf16/TF32 (`torch.backends.cudnn.allow_tf32`),
  and `channels_last`. Compare grouped-conv against `torch.func.vmap` for the conv layers.
   Check that actions don't change much under bf16, because argmax policies can be sensitive.
   Numerical regression tests disable TF32; training currently uses PyTorch defaults.
- **Weight loading**: batch `policy.load` for many slots at once instead of one slot at a time.
- **Test episodes**: 200 elite episodes, each up to the full ALE episode limit, are expensive. Consider testing
  every N generations, or with fewer episodes, during Colab runs.

## Phase 3: Colab usability

- [x] Add `colab/deep_ga_atari.ipynb`: repository clone, Drive mount, install, GPU
  diagnostics, tests, smoke/resume checks, optional pretrained replay, training, inline
  video, and saved command logs. Notebook structure checked locally; Colab execution pending.
- [ ] Add learning curves from `progress.csv` to the notebook.
- Periodic video of the current elite (every N generations) saved next to the snapshot.
- Color video: use a separate `gymnasium.make("ALE/<Game>-v5", render_mode="rgb_array")`
  replay driven by the same actions, or ALE's RGB screen, instead of the 84x84 grayscale input.
- Keep `snapshot.pkl` small (genomes are stored as seeds, alongside training/config/RNG
  state, rather than weight tensors); optionally keep per-generation copies.

Completed during the compatibility review: checkpoints now also save config/RNG state;
resume rejects incompatible configs, best-policy selection uses the elite's own score,
replay retains environment options, and MP4 encoding streams frames to bound host RAM.

## Phase 4: fidelity and remaining features of the original chapter

- Reproduce a published data point: run Frostbite with `ga_atari_config.json` for a
  fraction of the budget and compare the learning curve with the paper and with the
  original `snapshots/` output format (`master_extract_parent_ga` / `master_extract_cloud_ga`
  are not ported yet; add them if the book's visualization scripts need them).
- Port `es.py` + `optimizers.py` (OpenAI-ES with Adam) on top of the same evaluator and
  noise table.
- Port the hard-maze environment (`gym_tensorflow/maze`) as a NumPy/PyTorch vectorized
  environment, then add novelty search (GA-NS), which the original README lists as planned.
- Random-search baseline run (`rs_atari_config.json`) for comparison.

## Phase 5 (optional): JAX variant

- ale-py 0.12 ships an XLA vector environment that can run on the GPU (`ale-py[xla]`).
  Together with `evosax`/EvoJAX, this could remove the host round trip on every step.
  Worth prototyping only if phase 2 shows the CPU↔GPU exchange is the limit.

## Open questions

- Which Colab GPU/CPU combination is available to you (check `nproc` on the runtime)?
  The CPU core count decides whether `num_envs` above 256 helps.
- Should the default scoring stay with the original's no sticky actions
  (`repeat_action_probability=0`), or use the modern ALE v5 benchmark protocol (0.25)?
