import math
import numbers
import time

import numpy as np
import torch


class SharedNoiseTable:
    """A large fixed table of Gaussian noise that genomes index into.

    The table is generated with ``np.random.RandomState(seed).randn`` exactly like the
    original Uber implementation, so seeds produced by the original code (for example
    the Frostbite elite in ``pretrained/``) map to the same noise here. It is generated
    in chunks on the CPU (identical stream, bounded memory) and then kept on ``device``
    so that weights can be rebuilt from seeds directly on the GPU.
    """

    def __init__(self, count=250_000_000, seed=123, device="cpu", chunk=10_000_000):
        if count <= 0 or chunk <= 0:
            raise ValueError("Noise table count and chunk must be positive")
        t0 = time.time()
        print(f"Sampling {count} random numbers with seed {seed}")
        rs = np.random.RandomState(seed)
        host = np.empty(count, dtype=np.float32)
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            host[start:stop] = rs.randn(stop - start)
        self.noise = torch.from_numpy(host).to(device)
        print(f"Sampled {self.noise.numel() * 4} bytes in {time.time() - t0:.1f}s on {device}")

    def __len__(self):
        return self.noise.numel()

    def get(self, i, dim):
        if i < 0 or dim <= 0 or i + dim > len(self):
            raise ValueError(f"Noise slice [{i}:{i + dim}] exceeds table size {len(self)}")
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        if dim > len(self):
            raise ValueError(f"noise_table_size must be at least the policy parameter count ({dim})")
        return int(stream.randint(0, len(self) - dim + 1))


class ConstantSchedule:
    def __init__(self, value):
        self._value = value

    def value(self, **kwargs):
        return self._value


class LinearSchedule:
    def __init__(self, schedule, final_p, initial_p, field):
        self.schedule = schedule
        self.field = field
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, **kwargs):
        assert self.field in kwargs, f"Argument {self.field} not provided to scheduler. Available: {kwargs}"
        fraction = min(float(kwargs[self.field]) / self.schedule, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExponentialSchedule:
    def __init__(self, initial_p, final_p, schedule, field):
        self.linear = LinearSchedule(
            initial_p=math.log(initial_p),
            final_p=math.log(final_p),
            schedule=schedule,
            field=field)

    def value(self, **kwargs):
        return math.exp(self.linear.value(**kwargs))


_SCHEDULES = {
    "ConstantSchedule": ConstantSchedule,
    "LinearSchedule": LinearSchedule,
    "ExponentialSchedule": ExponentialSchedule,
}


def make_schedule(args):
    if isinstance(args, numbers.Number):
        return ConstantSchedule(args)
    kwargs = {key: value for key, value in args.items() if key != "type"}
    return _SCHEDULES[args["type"]](**kwargs)
