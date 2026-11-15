# Institute GPU execution

## Before launching jobs

- Rotate any password previously exposed. Do not paste the replacement into code, chat, `.yaml`, or Git.
- Connect using your institute-approved SSH key/password process. A private IP may require institute Wi-Fi or VPN.
- Read scheduler/allocation rules. Do not assume exclusive use or occupy every GPU.
- Check the allocated device with `nvidia-smi`; check disk with `df -h` and the scheduler with the institute's instructions.
- Run `uv run shapeguard doctor` in an isolated environment. No shared CUDA driver/toolkit changes are necessary for a compatible PyTorch wheel.

## Transfer

Clone this repository on the server after you have published it, or transfer the working source directory excluding `.venv`, caches and credentials. Keep `data/raw` and `data/prepared` relative layout intact. Do not copy a macOS virtual environment to Linux.

Install uv according to institute policy, then run:

```bash
uv sync --extra dev
uv run shapeguard doctor
uv run pytest
uv run shapeguard smoke --device cuda
```

Use the scheduler's assigned GPU environment. If an explicit device selection is required, set `CUDA_VISIBLE_DEVICES` according to the allocation, not arbitrary available indices.

For long SSH sessions use a keepalive-enabled, institute-approved connection or a persistent job/session mechanism. A lost monitoring connection must not trigger a second GPU process: verify ownership and identity of the recorded process first. The study records an active status and can resume checkpoints; never stop other users' jobs to recover yours.

## Pilot

Copy a config, explicitly set `device: cuda`, and choose batch size from measured VRAM. Start with float32; set `train.amp: true` only after CUDA smoke tests pass. Measure one complete epoch and validation before budgeting the study. Defaults are deliberately conservative (batch 2, widths 16/32/64/128/256).

Launch jobs through the scheduler or approved persistent terminal session. Commands print JSON training progress and save resumable checkpoints each epoch. Stop cleanly between epochs where possible. Resume revalidates data/configuration.

## Reproducibility

Freeze annotations before comparative runs. Use the same CUDA environment for latency comparisons. Do not compare Mac MPS latency against server CUDA latency as if they were architectural differences. Additional seeds measure training variability; bootstrap intervals measure conditional test-sample variability, a different source of uncertainty.

Raw/annotation data are not committed. The main binary study uses the original fracture masks, so no derived bone annotations are required. Keep relative source/prepared paths intact when transferring manifests. Keep any access tokens external to the repository.

On the approved shared A100 allocation, run one sequential study process with CUDA AMP and `cuda_memory_fraction: 0.075`. This caps the PyTorch allocator at approximately 3 GB; driver/context allocations are additional. Do not launch other training processes concurrently. Shared-GPU timing must be interpreted as variable rather than a clean speed benchmark.
