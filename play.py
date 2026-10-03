"""Replay an evolved policy and record a video (port of the original display.py).

    python play.py --run out/frostbite                     # best genome of a training run
    python play.py --seeds pretrained/frostbite_uber_elite.json --episodes 5
"""

import argparse
import json
import os
from contextlib import nullcontext

import imageio.v2 as imageio
import numpy as np
import torch

from neuroevolution import AtariEvaluator, BatchedPolicy, SharedNoiseTable
from ga import resolve_device


def load_genome(path):
    with open(path) as f:
        data = json.load(f)
    raw = data["seeds"]
    if raw is None:
        raise SystemExit(f"{path} does not contain a genome yet")
    seeds = (int(raw[0]),) + tuple((int(idx), float(power)) for idx, power in raw[1:])
    return data["game"], data["model"], seeds, data.get("env_kwargs", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Training output directory containing best.json")
    source.add_argument("--seeds", help="JSON file with game, model and seeds")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--video", default=None, help="MP4 path for the first episode (default: next to the genome)")
    parser.add_argument("--no-video", action="store_true", help="Report scores without recording video")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed for the no-op starts")
    parser.add_argument("--max-steps", type=int, default=27_000)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")

    path = os.path.join(args.run, "best.json") if args.run else args.seeds
    game, model, seeds, env_kwargs = load_genome(path)
    video = args.video or os.path.splitext(path)[0] + f"_{game}.mp4"
    print(f"{game} / {model}: genome with {len(seeds) - 1} mutations")

    device = resolve_device(args.device)
    evaluator = AtariEvaluator(game, 1, env_kwargs=env_kwargs)
    try:
        replay(evaluator, model, seeds, device, args, video)
    finally:
        evaluator.close()


def replay(evaluator, model, seeds, device, args, video):
    policy = BatchedPolicy(model, evaluator.num_actions, 1, device=device, obs_shape=evaluator.obs_shape)
    # The table is a prefix of one fixed random stream, so it only has to cover the largest index used.
    needed = max([seeds[0]] + [idx for idx, _ in seeds[1:]]) + policy.num_params
    noise = SharedNoiseTable(count=needed, device=device)
    policy.load(0, policy.compute_weights_from_seeds(noise, seeds), seeds)
    del noise

    env = evaluator.env
    obs, _ = env.reset(seed=args.seed)
    scores = []
    if not args.no_video:
        os.makedirs(os.path.dirname(os.path.abspath(video)), exist_ok=True)
    writer = nullcontext(None) if args.no_video else imageio.get_writer(video, fps=15, macro_block_size=1)
    # Stream frames to ffmpeg: a 27,000-step episode otherwise holds ~3 GB in RAM.
    with writer as video_writer:
        for episode in range(args.episodes):
            total, steps = 0.0, 0
            while steps < args.max_steps:
                if episode == 0 and video_writer is not None:
                    video_writer.append_data(np.repeat(np.repeat(obs[0, -1], 4, axis=0), 4, axis=1))
                action = policy.act(torch.from_numpy(obs)).cpu().numpy()
                obs, rew, terminated, truncated, _ = env.step(action)
                total += float(rew[0])
                steps += 1
                if terminated[0] or truncated[0]:
                    break
            else:
                obs, _ = env.reset()
            scores.append(total)
            print(f"Episode {episode + 1}: reward {total:.0f} after {steps} steps")

    print(f"Mean reward over {len(scores)} episode(s): {np.mean(scores):.1f}")
    if not args.no_video:
        print(f"Video: {video}")


if __name__ == "__main__":
    main()
