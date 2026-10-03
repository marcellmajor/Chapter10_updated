import math

import numpy as np
import torch
import torch.nn.functional as F

# Architectures from the original neuroevolution/models/dqn.py.
# ("conv", kernel, stride, filters, std) / ("dense", units, std)
ARCHITECTURES = {
    "Model": [
        ("conv", 8, 4, 16, 1.0),
        ("conv", 4, 2, 32, 1.0),
        ("dense", 256, 1.0),
        ("dense", None, 0.1),
    ],
    "LargeModel": [
        ("conv", 8, 4, 32, 1.0),
        ("conv", 4, 2, 64, 1.0),
        ("conv", 3, 1, 64, 1.0),
        ("dense", 512, 1.0),
        ("dense", None, 0.1),
    ],
}


def _same_padding(size, kernel, stride):
    """TensorFlow "SAME" padding: output = ceil(size / stride), extra pad on the far side."""
    out = math.ceil(size / stride)
    total = max((out - 1) * stride + kernel - size, 0)
    return out, total // 2, total - total // 2


class BatchedPolicy:
    """DQN-style policy network holding one independent weight set per environment slot.

    All slots are evaluated in a single pass: convolutions run as grouped convolutions
    (one group per slot) and dense layers as batched matrix multiplies. Genomes are
    flat parameter vectors laid out exactly like the original TensorFlow model
    (conv kernels as [k, k, in, out], dense as [in, out], NHWC flatten order), so that
    ``compute_weights_from_seeds`` reproduces the original weights for a given seed list.
    """

    def __init__(self, architecture, num_actions, num_slots, device="cpu", obs_shape=(4, 84, 84)):
        self.architecture = architecture
        self.num_actions = num_actions
        self.num_slots = num_slots
        self.device = torch.device(device)
        self.obs_shape = tuple(obs_shape)
        self.seeds = [None] * num_slots

        self.layers = []
        scale_by = []
        offset = 0
        channels, height, width = self.obs_shape
        flat = None
        spec = ARCHITECTURES[architecture]
        for i, layer in enumerate(spec):
            if layer[0] == "conv":
                _, kernel, stride, filters, std = layer
                out_h, top, bottom = _same_padding(height, kernel, stride)
                out_w, left, right = _same_padding(width, kernel, stride)
                tf_shape = (kernel, kernel, channels, filters)
                entry = dict(kind="conv", kernel=kernel, stride=stride, cin=channels, cout=filters,
                             pad=(left, right, top, bottom), tf_shape=tf_shape)
                fan_in = kernel * kernel * channels
                channels, height, width = filters, out_h, out_w
            else:
                _, units, std = layer
                units = num_actions if units is None else units
                if flat is None:
                    flat = channels * height * width
                tf_shape = (flat, units)
                entry = dict(kind="dense", cin=flat, cout=units, tf_shape=tf_shape)
                fan_in = flat
                flat = units

            w_size = int(np.prod(tf_shape))
            entry.update(w_offset=offset, w_size=w_size, b_offset=offset + w_size, b_size=entry["cout"],
                         relu=i < len(spec) - 1)
            offset += w_size + entry["cout"]
            # dqn.Model.create_weight_variable: std / sqrt(fan_in); biases start at zero.
            scale_by.append(np.full(w_size, std / np.sqrt(fan_in), dtype=np.float32))
            scale_by.append(np.zeros(entry["cout"], dtype=np.float32))
            self.layers.append(entry)

        self.num_params = offset
        self.scale_by = torch.from_numpy(np.concatenate(scale_by)).to(self.device)

        self.weights, self.biases = [], []
        for entry in self.layers:
            if entry["kind"] == "conv":
                k = entry["kernel"]
                w = torch.zeros(num_slots, entry["cout"], entry["cin"], k, k, device=self.device)
            else:
                w = torch.zeros(num_slots, entry["cin"], entry["cout"], device=self.device)
            self.weights.append(w)
            self.biases.append(torch.zeros(num_slots, entry["cout"], device=self.device))

    def describe(self):
        lines = [f"{self.architecture}: input {self.obs_shape}, {self.num_actions} actions, "
                 f"{self.num_params} parameters, {self.num_slots} slots on {self.device}"]
        for entry in self.layers:
            lines.append(f"  {entry['kind']:5s} {entry['cin']} -> {entry['cout']}  tf_shape={entry['tf_shape']}")
        return "\n".join(lines)

    # ---- genome <-> weights -------------------------------------------------

    def randomize(self, rs, noise):
        return (noise.sample_index(rs, self.num_params),)

    def compute_mutation(self, noise, parent_theta, idx, mutation_power):
        return parent_theta + mutation_power * noise.get(idx, self.num_params)

    def compute_weights_from_seeds(self, noise, seeds, cache=None):
        if cache:
            cache_seeds = [o[1] for o in cache]
            if seeds in cache_seeds:
                return cache[cache_seeds.index(seeds)][0]
            if seeds[:-1] in cache_seeds:
                theta = cache[cache_seeds.index(seeds[:-1])][0]
                return self.compute_mutation(noise, theta, *seeds[-1])
            if len(seeds) != 1:
                raise NotImplementedError("Genome is not a direct child of a cached parent")
        theta = noise.get(seeds[0], self.num_params) * self.scale_by
        for idx, power in seeds[1:]:
            theta = self.compute_mutation(noise, theta, idx, power)
        return theta

    # ---- slots --------------------------------------------------------------

    @torch.no_grad()
    def load(self, slot, theta, seeds):
        if self.seeds[slot] == seeds:
            return False
        theta = theta.to(self.device)
        for entry, w, b in zip(self.layers, self.weights, self.biases):
            flat_w = theta[entry["w_offset"]:entry["w_offset"] + entry["w_size"]].view(entry["tf_shape"])
            if entry["kind"] == "conv":
                flat_w = flat_w.permute(3, 2, 0, 1)
            w[slot].copy_(flat_w)
            b[slot].copy_(theta[entry["b_offset"]:entry["b_offset"] + entry["b_size"]])
        self.seeds[slot] = seeds
        return True

    @torch.no_grad()
    def forward(self, obs):
        """obs: uint8 tensor [num_slots, C, H, W] -> action scores [num_slots, num_actions]."""
        slots = obs.shape[0]
        x = obs.to(self.device, torch.float32).mul_(1.0 / 255.0)
        flattened = False
        for entry, w, b in zip(self.layers, self.weights, self.biases):
            if entry["kind"] == "conv":
                x = F.pad(x, entry["pad"])
                _, c, h, wd = x.shape
                k = entry["kernel"]
                x = F.conv2d(x.reshape(1, slots * c, h, wd),
                             w.reshape(slots * entry["cout"], c, k, k),
                             stride=entry["stride"], groups=slots)
                x = x.view(slots, entry["cout"], x.shape[-2], x.shape[-1]) + b[:, :, None, None]
            else:
                if not flattened:
                    x = x.permute(0, 2, 3, 1).reshape(slots, -1)
                    flattened = True
                x = torch.baddbmm(b.unsqueeze(1), x.unsqueeze(1), w).squeeze(1)
            if entry["relu"]:
                x = torch.relu(x)
        return x

    def act(self, obs):
        return self.forward(obs).argmax(dim=-1)
