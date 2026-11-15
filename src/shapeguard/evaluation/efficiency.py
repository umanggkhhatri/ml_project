import statistics
import time

import torch
from torch import nn

from shapeguard.reproducibility import environment, synchronize


def benchmark(model, device, size, repeats=30):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    model.eval()
    sample = torch.zeros((1, 1, size, size), device=device)
    macs = [0]
    handles = []

    def count(module, inputs, output):
        kernel = module.kernel_size[0] * module.kernel_size[1]
        macs[0] += output.numel() * kernel * module.in_channels // module.groups

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(count))
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    with torch.inference_mode():
        for _ in range(5):
            model(sample)
        synchronize(device)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            model(sample)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    parameters = sum(p.numel() for p in model.parameters())
    return {
        "parameters": parameters, "float32_parameter_bytes": parameters * 4,
        "conv_macs": macs[0], "conv_flops": 2 * macs[0],
        "flops_note": "Convolution-only estimate, 2 FLOPs/MAC; excludes normalization, interpolation, activations and attention products.",
        "latency_median_ms": statistics.median(timings), "latency_mean_ms": statistics.mean(timings),
        "latency_p95_ms": sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))],
        "batch_size": 1, "input_size": size, "device": str(device), "repeats": repeats,
        "timing_scope": "model only, float32, warmed up and synchronized",
        "environment": environment(device),
    }
