# Benchmark Execution Guide

This document describes how the current implementation runs benchmarks and
handles results, failures, and cleanup. For installation commands and benchmark
names, start with the [README](../README.md).

## Code map

| Location | Responsibility |
|----------|----------------|
| [NVIDIA_runner.py](../NVIDIA_runner.py), [AMD_runner.py](../AMD_runner.py) | Platform-specific entry points into the shared CLI |
| [Infra/cli.py](../Infra/cli.py) | Argument validation, hardware discovery, benchmark dispatch, output directories, final status |
| [Infra/runtime.py](../Infra/runtime.py) | Logging, subprocess execution, errors, timeouts, container and temporary-file cleanup |
| [Infra/tools.py](../Infra/tools.py) | Shared configuration validation, result formatting, output paths, container mounts |
| [Benchmarks/](../Benchmarks/) | Benchmark-specific build, execution, and measurement parsing |
| [Infra/fio.py](../Infra/fio.py) | Shared NVIDIA/AMD storage benchmark |
| [config.json](../config.json) | Model, pretraining, and selected benchmark settings |
| [install-dependencies.sh](../install-dependencies.sh) | Platform detection and dependency installation |

## CLI options

Run from the repository root. Both runners accept one or more benchmark names
or `all`, followed by options:

```bash
python3 NVIDIA_runner.py hbm nccl --verbose
python3 AMD_runner.py hbm transfer --output-dir /mnt/benchmark-results
python3 NVIDIA_runner.py all --continue-on-error --timeout 1800
```

| Option | Behavior |
|--------|----------|
| `--help` | Show available benchmarks and options |
| `--verbose` | Include commands and subprocess output |
| `--quiet` | Hide progress; keep results, errors, and final status |
| `--output-dir PATH` | Choose the parent directory for results |
| `--continue-on-error` | Run remaining benchmarks after a failure |
| `--timeout SECONDS` | Limit each external command; unlimited by default |

By default, a failure skips remaining benchmarks. See
[Commands, timeouts, and failures](#commands-timeouts-and-failures) for exit
codes and failure handling, and [Output directories and persistence](#output-directories-and-persistence)
for result locations and export rules.

## Execution lifecycle

1. **Validate arguments.** Names are case-insensitive. Repeated names are removed
   while preserving their first occurrence. `all` selects the platform's built-in
   benchmark order, rather than the order of additional arguments.
2. **Reserve an output folder and configure logging.** A single benchmark uses
   its own folder; multiple benchmarks first reserve a suite folder.
3. **Discover hardware.** NVIDIA uses the first GPU name reported by
   `nvidia-smi`. AMD requires `rocminfo` output containing `gfx942` and saves the
   raw output in `system_specs.txt`.
4. **Load and execute benchmarks sequentially.** Implementations are imported
   lazily. The CLI changes to the repository root before executing each
   benchmark and restores the caller's working directory afterward.
5. **Record outcomes and finalize.** Results appear as they become available;
   statuses, durations, and the final summary are saved at the end.

`--help` and invalid benchmark names are handled before hardware discovery or
optional GPU/container imports. A setup failure prevents benchmarks from running.
There is no parallel benchmark scheduler.

Benchmark methods run inside named stages such as `build`, `run`, or
`download models`. Not every benchmark has a build method; some perform all work
in `run`. AMD container benchmarks also have a managed container lifecycle.
The build stage announces "checking build requirements" rather than implying
compilation is necessary. NCCL explicitly reports "test build skipped" when it
reuses an existing binary, or "Building NCCL AllReduce test" when it invokes make.

Hardware detection is not a full compatibility check. In particular,
`nvidia-smi` can list GPUs whose CUDA contexts cannot initialize. The CLI
explicitly skips NVIDIA HBM on GB200 and restricts Llama pretraining to H200/GB200;
other restrictions may surface in the underlying tool.

## Configuration and dependencies

[config.json](../config.json) is not a universal configuration API for every
benchmark. Some implementations use fixed defaults or constructor arguments.

- LLM models must be enabled with `"use_model": true` and have the matching
  `"type": "nvidia"` or `"type": "amd"`.
- Shared LLM validation requires nonempty, equally sized input/output length
  lists, positive integer request/parallelism settings, and an enabled model.
  The exact field names differ between NVIDIA and AMD.
- Both AMD and NVIDIA HBM use five measurements and a ten-second delay,
  hardcoded in their implementations; there is no HBM configuration section.
- Pretraining's resolved host `mount_path` must contain the configured training
  script and `config.json`. Container registry and model access are separate
  prerequisites.
- AMD GEMM constructor arguments `i` and `w` control timed and warmup iterations.
  Direct shell execution requires `M N K iterations warmup_iterations`.

Malformed JSON, missing sections, invalid section types, and configuration I/O
errors are reported with the configuration path and section where possible.
Individual implementations remain responsible for benchmark-specific validation.

The installer detects AMD before NVIDIA if both discovery tools succeed.
It anchors its working directory to the script directory and warns when no
virtual environment is active.

| Profile | Installation behavior |
|---------|-----------------------|
| Common | [requirements/common.txt](../requirements/common.txt) supplies shared packages |
| NVIDIA, non-Blackwell | Common and [NVIDIA Torch](../requirements/torch-nvidia.txt) requirements are resolved together, then [FlashAttention](../requirements/flashattn.txt) is installed without build isolation or dependency resolution |
| NVIDIA, Blackwell | Common and [Blackwell Torch](../requirements/torch-blackwell.txt) requirements are installed; this branch does not install the NVIDIA FlashAttention profile |
| AMD | [ROCm Torch](../requirements/torch-amd.txt) is installed separately from common/build requirements, followed by the selected Triton and FlashAttention source revisions |

[requirements/build.txt](../requirements/build.txt) holds shared build tools.
AMD and Blackwell Torch remain unpinned; the profiles are not complete lockfiles.
The installer finishes with `pip check`. Dependency installation does not prove
that GPUs, containers, model permissions, or benchmarks are functional.

## Output directories and persistence

An example multi-benchmark invocation produces:

```text
Outputs/
  suite_run_1_20261019T120000Z/
    setup.log
    run.log
    status.json
  hbm_run_1_20261019T120000Z/
    hbm.log
    summary.md
    status.json
  nccl_run_1_20261019T120000Z/
    nccl.log
    summary.md
    status.json
```

A single benchmark keeps `setup.log`, `run.log`, and its results together in the
benchmark folder. Raw output, plots, and any NCCL build artifacts also live in
the selected output location.

### Numbering rules

- The default parent is `Outputs` under the caller's working directory.
  `--output-dir PATH` selects a different parent and its counters.
- Each benchmark has an independent counter; the same benchmark name shares
  that counter across platforms. Suites have their own counter.
- The next number is the highest existing matching directory number plus one,
  not the number of directories. Legacy timestamp/UUID folders are ignored.
- One invocation timestamp is formatted in UTC as `YYYYMMDDTHHMMSSZ`.
- A POSIX file lock on `.run.lock` serializes allocation so concurrent CLI
  invocations reserve distinct numbers. Do not remove the lock file while runs
  are active.
- Failed and interrupted attempts retain their folders. Benchmarks never
  attempted after an earlier failure receive no individual folder.

The suite's `run.log` combines diagnostics from setup and every attempted
benchmark. Each benchmark's log contains its own diagnostics. Results use
scoped output paths, so container mounts, FIO temporary files, model output,
and pretraining artifacts follow the selected benchmark directory.

### Status files

`summary.md` contains only measurements and is published only after a benchmark
passes. Results are buffered during execution and written through a temporary
file before being atomically moved into place. Failed, skipped, and interrupted
benchmarks do not export partial Markdown; a passing benchmark with no exported
measurements also creates no Markdown file. Suite folders do not get Markdown.

Run status remains in the console, diagnostic logs, and JSON files, not Markdown.
`status.json` contains the platform, overall status, exit code, and rows such as:

```json
{
  "benchmark": "hbm",
  "status": "PASS",
  "duration_s": 12.345,
  "output_dir": "/path/to/Outputs/hbm_run_1_20261019T120000Z"
}
```

Suite status includes all selected benchmarks. Each attempted benchmark also
gets a local status file containing only its row. A later benchmark can have a
successful local status even if an earlier failure makes the suite fail.

An unattempted benchmark has `status: "SKIPPED"`, `duration_s: null`, and
`output_dir: null`. An attempted but unsupported benchmark can also be
`SKIPPED`, but has a duration and output folder.

Status files are written during normal finalization, including handled failures
and Ctrl+C. A hard kill, host crash, or storage error can leave partial artifacts
or missing status files. A persistence failure is reported and must not turn a
successful measurement into an overall success message.

## Console output and logging

| Destination | What appears |
|-------------|--------------|
| stdout | PrettyTable tables without titles/descriptions, or plain-text measurements, then the final status summary |
| stderr, default | Concise progress and errors |
| stderr, `--verbose` | Progress, commands, subprocess output, and other DEBUG diagnostics |
| stderr, `--quiet` | Errors; results and final status still appear on stdout |
| Diagnostic logs | DEBUG-level details regardless of console verbosity |
| `summary.md` | Markdown measurements rather than PrettyTable borders |

[export_markdown](../Infra/tools.py) is the shared result publication path. During
CLI execution, it buffers Markdown until the benchmark passes and prints results
immediately with flushing. For table results, only the PrettyTable is printed;
titles and descriptions are retained in Markdown and diagnostic logs. Text-only
results retain their text. Existing PrettyTable objects supply their own table
layout. Console results are not delayed
until completion and can include partial measurements from a subsequently failed
benchmark. Direct calls outside the CLI's deferred-export context append Markdown
immediately.

Diagnostic records include benchmark/stage context, commands, working
directories, elapsed time, exit codes, and detailed failure tracebacks.

Console results and diagnostic formatting redact recognized secret environment
values and token/password patterns. This is best-effort protection, not a
guarantee that every artifact is sanitized: raw output files and Markdown
summaries should not be assumed to be scrubbed. Never pass credentials in
command arguments or publish logs without checking them.

## Commands, timeouts, and failures

[run_command](../Infra/runtime.py) starts commands in a new process session and
reads stdout/stderr concurrently. Output is logged while the process runs.
Explicitly captured output uses spooled temporary files; larger captures spill
to disk before being returned to the caller.

- Commands are checked. A nonzero exit raises `CommandError`; `check=False` is
  rejected rather than permitting unchecked benchmark execution.
- A missing executable raises `PrerequisiteError`.
- Shell commands run through `bash -o pipefail`, so a successful final pipeline
  command cannot hide an earlier pipeline failure.
- `--timeout SECONDS` must be finite and positive. It limits each external
  command, not the whole benchmark, suite, Python code, or approval prompt.
  There is no command timeout by default.
- Timeout or interruption initiates termination of the owned process group,
  escalating when necessary. This is not a GPU reset or a machine-wide process
  cleanup.
- Builds may remain silent while compilers or linkers are active. The runtime
  does not infer a hang from lack of output or provide automatic build
  heartbeats.

### Error categories

| Error | Meaning |
|-------|---------|
| `ConfigurationError` | Invalid configuration or failure reading it |
| `PrerequisiteError` | Missing executable, Python dependency, or other prerequisite |
| `UnsupportedHardwareError` | Explicitly unsupported hardware; reported as SKIPPED |
| `OutputError` | Missing, incomplete, or invalid measurements |
| `CommandError` | External command failure |
| `CommandTimeoutError` | Command exceeded its time limit |

These share the `BenchmarkError` base class. Expected failures produce concise
console errors and detailed logs; unexpected exceptions are identified as
internal errors with tracebacks in the logs.

By default, the first failure stops further benchmarks. `--continue-on-error`
allows remaining benchmarks to run but does not clear the suite's failure.
Ctrl+C stops the suite even with continuation enabled.

| Exit code | Overall outcome |
|-----------|-----------------|
| `0` | No failures: PASSED if any benchmark passed, otherwise SKIPPED |
| `1` | Benchmark, setup, or result-persistence failure |
| `2` | Invalid CLI arguments |
| `130` | Interrupted |

## NCCL discovery and recovery

[NCCLBandwidth.py](../Benchmarks/NVIDIA/NCCLBandwidth.py) searches for an
installation with both `include/nccl.h` and a linkable `libnccl.so`.
A runtime-only installation with just `libnccl.so.2` is insufficient for building
the test.

Discovery checks explicit `NCCL_HOME`/`NCCL_ROOT` prefixes first, followed by
`LD_LIBRARY_PATH`-derived prefixes, standard locations (including multiarch library
directories), Python `nvidia/nccl` locations, and a complete local `nccl/build`.
The first matching installation is selected; finding files is not yet proof of
runtime compatibility.

When an installation is found:

1. The full NCCL library build is skipped.
2. Its header/library paths are passed through `NCCL_HOME`, compiler/linker flags,
   and `LD_LIBRARY_PATH`. The runtime version is queried from that specific
   library using `ncclGetVersion` in a separate process, then shown as an INFO
   message such as `Using NCCL 2.28.9`. This is the selected runtime version,
   not the header version used to compile a previously cached test binary.
3. The existing repository `nccl-tests` source is used if present; otherwise a
   checkout is cloned into `nccl-tests` at the repository root.
4. An existing `nccl-tests/build/all_reduce_perf` is reused without invoking
   `make`. If that executable is missing, `make` runs in `nccl-tests`, using its
   normal build directory.
5. The AllReduce benchmark runs against the selected library.

Repeated runs do not create per-run test build directories or automatically
invalidate the existing binary when library, toolkit, or source versions change.
This assumes a stable installation during the testing window. An initial test
build can still take time or require a source checkout.

### Rebuild approval

If test compilation or execution raises `CommandError` or measurement parsing
raises `OutputError` with the detected installation, an interactive terminal asks:

```text
Build NCCL from scratch and retry NCCL tests? [y/N]
```

- Only `y` or `yes` approves recovery.
- Declining, EOF, or noninteractive input fails without rebuilding.
- Approval clones fresh NCCL into `nccl-source` inside the benchmark output
  folder, builds it, forces the existing test build with `make -B` in the
  repository's `nccl-tests` directory, and retries once.
- Failed fresh builds or retries propagate as failures; there is no retry loop.
- If no development installation was found initially, a fresh source build is
  performed directly without this approval step, and the test is also forcibly
  rebuilt against that library.
- Ctrl+C does not trigger recovery. Missing prerequisites are not repaired by
  this mechanism.

The existing repository NCCL library tree is left untouched. Test build artifacts
are reused or rebuilt in their usual repository location; no `make clean` or
source reset is performed. The current implementation does not pin freshly
cloned NCCL or test revisions.

### GPU and measurement limitations

The test requests all GPUs counted by `nvidia-smi`, with message sizes from
`8` to `8G`, doubling each step, and `40` iterations. It uses Ring on four-GPU
systems and NVLS otherwise. It expects 31 bandwidth measurements and rejects
incomplete output instead of publishing an empty success table.

GPU counting is not a CUDA initialization probe and is not adjusted using
`CUDA_VISIBLE_DEVICES`. Do not assume a visibility mask reduces the requested
GPU count.

The recovery prompt does not establish the cause of a test failure. For
`CUDA-capable device(s) is/are busy or unavailable`, check GPU availability,
CUDA context initialization, driver/kernel errors, and fabric health before
approving a rebuild. Recompiling NCCL does not repair an unavailable GPU.

## Containers and storage safety

Container benchmarks require both the Docker CLI and SDK plus daemon
permissions. Runners do not automatically invoke sudo for Docker.

- Image pulls and container commands use the shared command runner.
- Container execution receives `BENCHMARK_OUTPUT_DIR`, and output directories
  are mounted at the corresponding host paths.
- Managed containers are stopped on success, failure, and interruption.
  Startup cleanup also attempts to remove an owned container if creation failed
  after allocating it.
- Cleanup errors are logged without replacing an earlier benchmark failure.
  Cleanup is best-effort; daemon outages or hard kills can prevent it.

[FIO](../Infra/fio.py) runs eight read/write cases in an isolated temporary
directory on the output filesystem. It uses four jobs with a configured
10G file size per job, so sufficient free space is required. It sums
`bw_bytes` across jobs and reports MiB/s. Empty job lists and nonzero job errors
fail the benchmark. Cleanup targets only its owned temporary directory.

## Testing and scope

Run the hardware-free suite from the repository root:

```bash
python3 -m unittest discover -s unit-tests -v
```

The [unit tests](../unit-tests/) stub optional GPU/container dependencies and external
commands. They cover command failures, logging, result rendering, directory
allocation, dependency installation flow, NCCL discovery/approval/retry, and
benchmark parsing regressions.

Passing these tests does not verify GPU kernels, Docker image compatibility,
model downloads, real performance, or end-to-end training. Those require
integration testing on the target machine.
