# MLPerf Inference Heterogeneous MIG Workloads
## Heterogeneous MIG Workloads for Multi-MIG Systems

In NVIDIA submission machines where we support [MIG (Multi-Instance GPU)](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/), NVIDIA's MLPerf Inference submission has multiple variants where we run inference on the entire GPU, a single 1-GPC MIG slice, and all available 1-GPC MIG Slices (Multi-MIG). In the Multi-MIG case, in addition to having results for when all MIG slices are processing the same benchmark, NVIDIA presents a use-case where each MIG slice can run different and unique benchmarks, 1 on each MIG slice. We call this a **Heterogeneous MIG Workload** (**HeteroMIG** or **HeteroMultiUse** for short).

In this heterogeneous setting, benchmarks are divided into a ***'main benchmark'***, which is the benchmark / model whose performance we are interested in measuring for this setting, and ***'background benchmarks'***, where we launch as uniform of a distribution as possible of MLPerf Inference benchmarks that are **not** the same model as the main benchmark. In fact, these background benchmarks do not even need to be the same MLPerf Inference *scenario* as the main benchmark: they are completely independent and agnostic, to show parallel computing capability.

### How the HeteroMIG harness is designed

1. First, the harness will launch the background benchmarks. These background benchmarks run with `min_duration` set to greater than the `min_duration` or expected runtime of the main benchmark. Furthermore, since this workload design is agnostic to the scenarios of the background benchmarks, all background benchmarks will run under the 'Offline' scenario to showcase the "worst-case scenario", as 'Offline' creates the most stress on the system.
2. After *all* background benchmarks report 'Running actual test' to signal that LoadGen has started and inference is being run, the harness will wait for a configurable period.
3. After this initial 'wait' period, the harness will launch the main benchmark.
4. Once the main benchmark has completed, the harness will either wait for the background benchmarks to complete or kill the background benchmarks.
5. The HeteroMIG harness will only report a successful run if the the main benchmark was a valid result, and **all** background benchmarks were completed or killed **after** the main benchmark completed.

### Supported Systems

We currently support HeteroMIG on A100-SXM-80GB (DGX A100, 80GB variant), A100-PCIe-80GB, and A30 configurations.

### Building the engines before running the harness

MIG by design only allows an individual CUDA process to "see" a single MIG slice. In order to target a specific MIG GPU instance on a system with multiple MIG slices instantiated, the `CUDA_VISIBLE_DEVICES` environment variable for that process must be set to the UUID of the MIG instance.

For example, if our targeted MIG slice has a UUID of `MIG-7fdeccba-6a8f-50a1-a833-e8e357d1bfd1`, to prepare all the engines for the HeteroMIG harness, run:

```
$ CUDA_VISIBLE_DEVICES=MIG-7fdeccba-6a8f-50a1-a833-e8e357d1bfd1 make generate_engines \
    RUN_ARGS="--scenarios=offline,server --benchmarks=resnet50,ssd-resnet34,bert --config_ver=hetero"
$ CUDA_VISIBLE_DEVICES=MIG-7fdeccba-6a8f-50a1-a833-e8e357d1bfd1 make generate_engines \
    RUN_ARGS="--scenarios=offline,server --benchmarks=dlrm,3d-unet,bert --config_ver=hetero_high_accuracy"
```
3d-unet and DLRM use 'hetero_high_accuracy' as the default scenario because both benchmarks have one engine to support both high and low accuracy targets. All other benchmarks use 'hetero' as default scenario, for low accuracy target. BERT has separate engine to build for 'hetero_high_accuracy'.

Internally, the HeteroMIG harness uses similar commands to run the harnesses, via the `run_harness` make target, behind the scene.

### Commands to run the HeteroMIG harness

The main script is located under `closed/NVIDIA/scripts/launch_heterogeneous_mig.py`. This script is a wrapper that runs and tracks multiple single-benchmark harnesses that run as the main and background benchmarks via `make run_harness`.

To see all command-line options this script supports, from inside the container, run:

```
python3 scripts/launch_heterogeneous_mig.py --help
```
The main command-line options to use are:

- `--main_benchmark` to set the main benchmark model
- `--main_scenario` to set the scenario for the main benchmark
- `--background_benchmarks` to set the list of models to use for the background benchmarks
    - Supports 'all', 'none', 'edge', and 'datacenter' as values. Default: 'all'.
    - **IMPORTANT**: Because this workload is designed to mimic a datacenter running multiple models for inference, we only submit HeteroMIG as a "datacenter"-type submission. As such, set `--background_benchmarks=datacenter` to use all 6 'datacenter'-type submission-required models for the background benchmarks.
- `--start_time_buffer` to configure the wait time to start running the main benchmark after all background benchmarks have started running their inferences (Units: ms, Default: 600000)
- `--background_benchmark_duration` to configure `min_duration` for the background benchmarks. If set to 'automatic', the script will automatically maintain that the background benchmarks keep running until the main benchmark finishes. Default: 'automatic'.

As examples, to run with SSDResNet34 under the server scenario as the main benchmark as a datacenter submission:

```
python3 scripts/launch_heterogeneous_mig.py \
    --main_action=run_harness --background_benchmarks=datacenter \
    --main_benchmark=ssd-resnet34 --main_scenario=server
```
Then to run the AccuracyMode for above workload:

```
python3 scripts/launch_heterogeneous_mig.py \
    --main_action=run_harness --background_benchmarks=datacenter \
    --main_benchmark=ssd-resnet34 --main_scenario=server \
    --main_benchmark_runargs="--test_mode=AccuracyOnly"
```
If you want to run AUDIT_TEST01 instead of a normal inference run:

```
python3 scripts/launch_heterogeneous_mig.py \
    --main_action=run_audit_test01 --background_benchmarks=datacenter \
    --main_benchmark=ssd-resnet34 --main_scenario=server
```
### Tuning performance and target QPS of the main benchmark

For tuning, it is recommended to focus on each offline and server scenario benchmark individually first as if they were running on a single MIG slice independently, with no other inference running on any GPU. In most cases, we achieve similar results between the single-MIG and HeteroMIG workloads.

As an example, DLRM, which is around 7% slower in the Offline scenario in HeteroMIG, was first tuned under the single-MIG setting to find a baseline for HeteroMIG tuning. The target QPS was then gradually lowered during the tuning process in HeteroMIG until the runs were no longer INVALID.

