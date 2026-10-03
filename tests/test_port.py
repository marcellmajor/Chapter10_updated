"""Numerical compatibility checks. Run with: python -m pytest tests/."""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroevolution import BatchedPolicy, SharedNoiseTable  # noqa: E402


@pytest.fixture
def full_precision():
    # Test numerical equivalence without Ampere+ TF32 rounding. Training uses
    # PyTorch's defaults and is separately exercised by the CUDA smoke command.
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def test_noise_table_matches_original_generation():
    table = SharedNoiseTable(count=1_234_567, seed=123, chunk=100_000)
    expected = np.random.RandomState(123).randn(1_234_567).astype(np.float32)
    np.testing.assert_array_equal(table.noise.numpy(), expected)


def _reference_forward(policy, theta, obs):
    """Single-network forward pass written directly against the original TF layout (NHWC)."""
    x = obs.float().div(255.0).permute(1, 2, 0).unsqueeze(0)  # 1 x H x W x C
    for entry in policy.layers:
        w = theta[entry["w_offset"]:entry["w_offset"] + entry["w_size"]].view(entry["tf_shape"])
        b = theta[entry["b_offset"]:entry["b_offset"] + entry["b_size"]]
        if entry["kind"] == "conv":
            nchw = F.pad(x.permute(0, 3, 1, 2), entry["pad"])
            x = F.conv2d(nchw, w.permute(3, 2, 0, 1), stride=entry["stride"]).permute(0, 2, 3, 1) + b
        else:
            x = x.reshape(1, -1) @ w + b
        if entry["relu"]:
            x = torch.relu(x)
    return x[0]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU not available"))])
def test_batched_forward_matches_reference(device, full_precision):
    torch.manual_seed(0)
    rs = np.random.RandomState(0)
    for arch in ("Model", "LargeModel"):
        policy = BatchedPolicy(arch, num_actions=18, num_slots=3, device=device)
        noise = SharedNoiseTable(count=policy.num_params + 1000, seed=1, device=device)
        thetas = []
        for slot in range(3):
            seeds = policy.randomize(rs, noise) + ((noise.sample_index(rs, policy.num_params), 0.01),)
            theta = policy.compute_weights_from_seeds(noise, seeds)
            policy.load(slot, theta, seeds)
            thetas.append(theta)
        obs = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
        batched = policy.forward(obs).cpu()
        for slot in range(3):
            ref = _reference_forward(policy, thetas[slot].cpu(), obs[slot])
            torch.testing.assert_close(batched[slot], ref, rtol=1e-4, atol=1e-5)


def test_parameter_counts():
    assert BatchedPolicy("Model", 18, 1).num_params == 1_008_450
    assert BatchedPolicy("LargeModel", 18, 1).num_params == 4_052_658


def test_cached_mutations_match_full_decode():
    policy = BatchedPolicy("Model", 3, 1)
    noise = SharedNoiseTable(count=policy.num_params + 100)
    parent = (1, (2, 0.002))
    theta = policy.compute_weights_from_seeds(noise, parent)
    child = parent + ((3, 0.002),)
    torch.testing.assert_close(policy.compute_weights_from_seeds(noise, child, cache=[(theta, parent)]),
                               policy.compute_weights_from_seeds(noise, child), rtol=0, atol=0)
