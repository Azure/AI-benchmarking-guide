# Benchmark Execution Guide

See the [README](../README.md) for prerequisites, model configuration, and
benchmark names.

## Running

From the repository root:

```bash
python3 NVIDIA_runner.py hbm nccl --verbose
python3 AMD_runner.py hbm transfer --output-dir /mnt/benchmark-results
python3 NVIDIA_runner.py all --continue-on-error --timeout 1800
```

- `--verbose` shows commands and subprocess output; full diagnostics are always
  saved to logs. `--quiet` hides progress but retains results and errors.
- `--timeout` limits **each external command**, not the whole benchmark.
  Commands have no timeout by default.
- Benchmarks run sequentially and stop at the first failure.
  `--continue-on-error` runs the remaining tests without clearing earlier failures.

## Results

Runs are saved under `Outputs/<benchmark>_run_<number>_<UTC timestamp>/`, or
the parent selected by `--output-dir`.

| File | Purpose |
|------|---------|
| `summary.md` | Measurements, published only after the benchmark passes |
| `status.json` | Final outcome, duration, exit code, and output paths |
| `<benchmark>.log` | Commands, subprocess output, and failure details |
| `run.log` | Combined diagnostics for the invocation |

Multi-benchmark runs also create a `suite_run_...` folder with combined logs and
status. Individual benchmarks retain their own results. Console tables can appear
before a later failure; use the final status rather than the presence of a table
to determine success. Hard kills can leave status files unfinished.

Review logs and artifacts before sharing them; secret redaction is best-effort.

## Benchmark-specific notes

| Benchmark | Caveat |
|-----------|--------|
| AMD | Requires MI300X (`gfx942`). |
| NVIDIA HBM | GB200 builds target `sm_100`; requires a CUDA toolkit supporting that architecture. |
| Llama pretraining | H200/GB200 only; `mount_path` must contain the training script and `config.json`. |
| AMD GEMM | First-run hipBLASLt compilation can take tens of minutes, with long silent periods. |
| AMD RCCL | Uses the container's installed library and builds only pinned AllReduce tests in the run directory. |
| FIO | Writes four 10G files in a temporary directory on the output filesystem; allow sufficient free space. |

### TransferBench matrix

TransferBench reports host-transfer bandwidth and a full GPU-pair matrix of
**bidirectional GPU-kernel read totals**, in GB/s. It requires at least two GPUs.

Each cell sums both simultaneous directions for one pair, using 32 CUs per
destination GPU. Pairs run separately, not all-to-all concurrently. Opposite
cells may differ because they are measured separately. `-` marks self-transfers
or unavailable peer paths. This is not a copy-engine measurement.

### NCCL discovery and recovery

NCCL needs development headers and a linkable `libnccl.so`, not just the runtime
`libnccl.so.2`. Set `NCCL_HOME` or `NCCL_ROOT` to select an installation.

Installed NCCL and existing `nccl-tests/build/all_reduce_perf` are reused.
Cached tests are **not automatically rebuilt** when toolkit or library versions
change.

If tests fail with the detected library, an interactive prompt offers one
source rebuild and retry. Only `y`/`yes` approves; noninteractive runs fail
without rebuilding. If no development installation is found initially, the
source build runs automatically. Fresh library sources go under the run's
`nccl-source`; test binaries remain in the repository's `nccl-tests/build`.

The test counts GPUs using `nvidia-smi`, **ignoring `CUDA_VISIBLE_DEVICES`**.
It selects Ring for four GPUs and NVLS otherwise. Check GPU availability before
approving a rebuild for CUDA initialization errors.

## Docker

AMD GEMM, RCCL, and FlashAttention reuse one image; LLM uses a separate image.
The shared image is set in [Infra/tools.py](../Infra/tools.py).

Images are pulled only when missing. Refresh a cached tag with `docker pull IMAGE`.
Docker requires both the CLI and Python SDK, with daemon permissions for the
calling user. Containers are cleaned up after runs; a hard kill or daemon outage
may prevent cleanup.

### TensorRT-LLM container setup

After entering the TensorRT-LLM container, install `prettytable` before running
the NVIDIA LLM benchmarks:

```bash
python3 -m pip install prettytable
```

### Relocating Docker and containerd storage

Docker's containerd image store uses a separate path from Docker's `data-root`.
Configure `data-root` as shown in the [README](../README.md#storage-and-docker).
Merge these **top-level** settings into `/etc/containerd/config.toml`, preserving
existing runtime settings and leaving `state` unchanged:

```toml
root = "/mnt/resource_nvme/containerd"
temp = "/mnt/resource_nvme/containerd/tmp"
```

For existing installations, back up both configs and stop services **before
editing or migrating**. Preserve existing Docker settings, including NVIDIA
runtimes. Substitute your actual mount and source paths if different:

```bash
sudo systemctl stop docker.service docker.socket containerd.service
# Edit the configs above, then migrate existing data (requires rsync).
sudo mkdir -p /mnt/resource_nvme/docker /mnt/resource_nvme/containerd
sudo rsync -aHAX --numeric-ids /var/lib/docker/ /mnt/resource_nvme/docker/
sudo rsync -aHAX --numeric-ids /var/lib/containerd/ /mnt/resource_nvme/containerd/
sudo mkdir -p /mnt/resource_nvme/containerd/tmp
sudo systemctl start containerd.service docker.socket docker.service
```

Ensure the destination disk is mounted before either service starts. Keep the
original data until pulls and container execution succeed.
