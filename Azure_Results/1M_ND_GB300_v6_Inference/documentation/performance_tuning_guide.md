# NVIDIA MLPerf Inference System Under Test (SUT) Performance Tuning Guide
This guide is meant to provide details on how to fix issues involving bad performance, "invalid results", or potential hard crashes after following the "Prepping our repo for your machine" section of the MLPerf Inference Tutorial (this is the same as the closed/NVIDIA/README.md stored in the repo).

After you add your system description to system_list.py and add configs for your desired benchmarks, it is possible there might be issues causing you to not achieve the best performance possible. Before diagnosing any possible errors, **please go through each of these sections in order, as some pertain to adhering to MLPerf Inference rules**.

**IMPORTANT:** Make sure your performance tuning changes (i.e. any change made following steps on this document) are done in `configs/[BENCHMARK]/[SCENARIO]/__init__.py` files. Note that all files in the `measurements/` directory are automatically generated from the files in the `configs/` directory at runtime, so any manual changes made in `measurements/` will not take effect.

### Identifying performance issues via nvidia-smi metrics
nvidia-smi provides the user, in real time, a list of key metrics that that reflects different aspects of GPU performance. For a comprehensive list of metrics, se `nvidia-smi --help-query-gpu`.

We provide a set of scripts to collect these metrics during benchmark runs. These scripts are at closed/NVIDIA/scripts/perf_monitor:
- `run_nvidia_smi_mon.py`: Collects specified metrics during a command run and saves them to an output CSV file. See `python3 run_nvidia_smi_mon.py --help` for usage.
- `plot_nvidia_smi_mon.py`: Takes as input the CSV file above and outputs a plot. This can be static png files or interactive HTML plots. See `python3 plot_nvidia_smi_mon.py --help` for usage.

For convenience, we also provide a make target, `make perf_profile` to generate static plots easily to identify performance issues or verify stability of performance.
The following arguments are required:
- `IMG_FILE`: The path to the output PNG file.
- `CSV_FILE`: The path to the nvidia-smi metrics CSV file.
- Either `COMMAND` or (`RUN_ARGS` and `ACTION`): To either run an arbitrary command, or to run an ACTION (which is a make target like `run_harness_power` or `run_harness`) with `RUN_ARGS` (specifying the scenarios and benchmarks)
Examples:
```
$ make perf_profile ACTION=run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline" CSV_FILE=out2.csv IMG_FILE=plot-bert.png
$ make perf_profile COMMAND="bash_command_here" CSV_FILE=out3.csv IMG_FILE=plot.png
```

### LLM Performance tuning
#### Batching strategies
There are 2 batching strategies:
- Static batching: Every request of a batch must complete before the next batch can be processed. Since the output sizes of LLM requests may be skewed, this suffers from low utilization, harming throughput.
- In-flight batching: Here, the batch of requests is dynamically adjusted, adding new inputs and removing completed ones in real-time, rather than using a fixed batch size. Keeps utilization high, and is good for throughput. It works as follows:
    - At each generation step, the model processes all active sequences.
    - Completed sequences (e.g., those that hit an end-of-text token) are removed.
    - New requests are injected into the batch immediately, without waiting for a full batch reset.

We use IFB for LLM workloads _on Datacenter systems_.

#### TRTLLM parameters
##### `max_num_tokens`
This is the maximum number of tokens processed in every iteration. Best way to tune this:
- Use whatever NVIDIA is using (for the same GPU)
- If we haven't published configs for the GPU used, set some default value (2048) and launch a run with logging enabled (`--verbose_glog=1` in RUN_ARGS). This dumps statistics at the end of every IFB iteration.
- (Perform this step only for the `Offline` scenario) From the dumped iteration logs, notice the "free KV cache blocks". We need to now tune the batch size such that this is minimized, but does not hit 0.
- (Perform this step for `Offline`/`Server` separately) Set max_num_tokens = (1 + isl/osl)*max_batch_size, and tune from there.

##### `kvcache_free_gpu_mem_fraction`
This parameter decides the GPU memory reserved for the KV cache. Is between 0 and 1. 
- Free GPU memory is defined as ((Total GPU memory) - (Model params size)). 
- Since benchmark runs are isolated, we set a high number (>0.9) – with just enough space for input/output.
- For larger memory (B200), we may go upto 0.95.

#### Parallelisms
Sometimes, it is required to split a single instance of the model between multiple GPUs (Llama3.1-405B).
- Tensor parallelism: Splits the model by tensors. Has a high communication overhead, but (generally) lower latency, lower throughput.
- Pipeline parallelism: Splits the model by layers. Has a low communication overhead, but (generally) higher throughput, higher latency.

#### [Chunking strategy](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)
When chunked context is enabled, this dictates how the context requests are scheduled.
- First Come First Serve: All context chunks of earlier requests will be selected before that of the next requests.
- Equal Progress: Chunks will be selected uniformly for all pending requests, irrespective of arrival time.
By default, we use First Come First Serve

### Using NSYS to inspect performance
NSYS profiles are useful in gaining insight as to where opportunities for performance optimizations lie. Please use the `make perf_profile` first to report performance issues, and capture nsys only when asked/required.
For non-LLM workloads:
```
nsys profile -f true --gpu-metrics-devices=all -y 270 –d 30 -o out.nsys-rep ​make run_harness RUN_ARGS="--benchmarks=sdxl –scenarios=Offline​"
```

For LLM workloads, instead of relying on the delay `-y` and duration `-d` flags, we can profile a range of the TRTLLM iterations, which is more useful:
```
TLLM_GPTM_PROFILE_START_STOP=2000-2050 nsys profile --output=out.nsys-rep --force-overwrite=true -t cuda,nvtx --gpu-metrics-devices=all -c cudaProfilerApi --capture-range-end="stop-shutdown" make run_harness RUN_ARGS="--benchmarks=llama2 --scenarios=offline"
```

### Different system configurations that use the same GPU configuration
Sometimes, it may be the case that submitters will have 2 different submission systems with the same GPU configuration, but differing hardware configurations elsewhere, such as a different CPU, memory size, etc. These are counted as separate systems, and you should have definitions for these which specify the CPUs and memory sizes, as well as different system IDs. This also means you will need separate benchmark configurations for each unique system. See the main README.md for instructions on how to add a system. If you are using the automated script, you will need to run it once on each system.

### Using the 'start_from_device' and 'end_on_device' flag

On some systems, it is possible that you can use the `start_from_device` and `end_on_device` flags to improve your end-to-end inference performance results. When `start_from_device` is enabled, the QSL is loaded on GPU memory, and any Host-to-Device memcpy is skipped. This behavior of this flag, however, **requires approval from the MLPerf Inference committee** and the submitter **must provide proof** that their system has a NIC directly attached to each GPU, and the data from the network can be inserted into device memory *without* going through the host (CPU). MLPerf Inference ***requires*** the following as proof:

- A system architecture diagram showing direct connection between the network and accelerator, without going through the CPU.
- Bandwidth measurements for this connection that show they are sufficient for the benchmark
- Example of software that supports this data connection (i.e. GPUDirect).

The 'minimum required bandwidth' is dependent on the benchmark and its input. You can only use `start_from_device` for benchmarks you can satisfy the minimum bandwidth requirement of. **If your system does not qualify for the requirements to use start_from_device, do not specify it in your BenchmarkConfiguration (i.e. delete the field).**

Formulas for minimum required bandwidth for **start_from_device** are as follows:

- start_from_device (in bytes), where tput is your throughput as 'Queries per second':
    - ResNet50: throughput x (C x H x W) = throughput x (3 x 224 x 224) = tput x 150528
    - SSDMobileNet: throughput x (C x H x W) = throughput x (3 x 300 x 300) = tput x 270000
    - SSDResNet34: throughput x (C x H x W) = throughput x (3 x 1200 x 1200) = tput x 4320000
    - BERT-99: throughput x ( 2 x max_seq_len + 1) x sizeof(int32) = throughput x ( 2 x 384 + 1) x 4 = tput x 3076
    - BERT-99.9: throughput x ( 2 x max_seq_len + 1) x sizeof(int32) = throughput x ( 2 x 384 + 1) x 4 = tput x 3076
    - DLRMv2: throughput x num_pairs_per_sample * (num_numerical_inputs_padded x sizeof(int8) + num_categorical_inputs x sizeof(int32)) = throughput x 270 * (16 x 1 + 26 x 4) = tput x 32400
        - Note that, numerical inputs are padded to 16B.
    - 3D-UNET: throughput x (inputC x D x H x W) = throughput x (1 x averageof(43 input tensor sizes)) = tput x 32944795
        - Note that, 3D-UNET KiTS19 input set has 43 inputs with 15 different shapes. The above size is from the average of 43 inputs (as opposed to the use of max tensor size for the worst case scenario). This is reasonable because benchmark requires to run whole inputs multiple times, i.e. multiple iterations of the entire 43 samples sorted in random order per iteration, to avoid skewed results from non-uniform selection of inputs.

There is also an `end_on_device` flag introduced in v1.1 of MLPerf Inference. This flag disables the Device-to-Host memcpy that is normally done post-inference. To use this flag, MLPerf Inference ***requires*** the submitter to provide proof of sufficient bandwidth to transmit the result back to the network directly from the device. **If your system does not qualify for the requirements to use end_on_device, do not specify it in your BenchmarkConfiguration (i.e. delete the field)**.

**Note that this flag is only supported / allowed on the following benchmarks (as per MLPerf Inference rules):**

- 3D-UNET

Formulas for minimum required bandwidth are as follows:

- end_on_device (in bytes), where tput is your throughput as 'Queries per second':
    - 3D-UNET: throughput x (outputC x D x H x W x sizeof(int8)) = throughput x (1 x averageof(43 input tensor sizes) x 1) = tput x 32944795
        - 3D-UNET KiTS19 output shows segmentation results (labels) of each voxel, showing if it is for background (=0), Normal cell (=1), or Tumor (=2). The segmentation results are in 1 channel INT8 dtype. The size of the output tensor is the same to input tensor.

**If you wish to use both start_from_device and end_on_device, you must provide the proof for bidirectional bandwidth.** Your system must be able to support the sum of the start_from_device **and** end_on_device bandwidth requirements.

The benchmarking philosophy of MLPerf is to include the transmission of data between the accelerator and the most logical placement of the data as part of the timed portion of the benchmarking task. This "most logical placement of data" is most commonly the CPU DRAM, in which the data is placed by an I/O device prior to preprocessing and running the timed portion of the benchmark, and then transmitted to an I/O device after the end of the timed portion and any postprocessing. The start_from_device and end_from_device exceptions cover cases where the system topology and usage scenario allow the data to never pass through CPU memory on the way to or from the I/O device to the accelerator. NVIDIA's interpretation of the rules is thus that **if the system topology requires data to pass through CPU memory**, start_from_device and end_on_device **are not legal optimizations**.

**Furthermore, in order to use start_from_device or end_on_device in a BenchmarkConfig, you must add the GPU to the enable-list.** The enable-lists are located in `code.common.systems.system_list.SystemClassifications`, under `start_from_device_enabled` and `end_on_device_enabled`, respectively.

### Using NUMA configurations

[NUMA (Non-Uniform Memory Access)](https://man7.org/linux/man-pages/man7/numa.7.html) is a feature for multi-processor systems to divide memory into multiple memory nodes. If your system has multiple CPUs and supports NUMA, then configuring NUMA correctly might be required for optimal performance. Our codebase allows users to specify NUMA configurations for specific systems via the `numa_config` key in the system's config block. **Note that some of our configurations do this already for certain system IDs. If you have a system with an identical GPU layout as an NVIDIA submission, or copied from an existing config block as a template, please REMOVE the numa_config key-value pair, as your system most likely does not have the same NUMA configuration as ours.**

To check if your system supports NUMA, you can use the `numactl` tool:

```
$ numactl --hardware

available: 4 nodes (0-3)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
node 0 size: 128825 MB
node 0 free: 124581 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
node 1 size: 129016 MB
node 1 free: 127519 MB
node 2 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
node 2 size: 129016 MB
node 2 free: 120735 MB
node 3 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 3 size: 128978 MB
node 3 free: 125733 MB
node distances:
node   0   1   2   3
  0:  10  12  12  12
  1:  12  10  12  12
  2:  12  12  10  12
  3:  12  12  12  10
```
The above example shows that there are 4 NUMA nodes (0-3). Each NUMA node has 1 or more CPU cores and a memory address space that is closer (latency-wise) to those CPUs compared to the CPUs of other NUMA nodes. For example, in the above sample output, CPU ID 2 is in NUMA node 0, so CPU 2 can access memory addresses of NUMA node 0 with less latency than memory addresses of NUMA node 3. This CPU may get higher memory bandwidth when accessing memory addresses in NUMA node 0's memory address space. The memory access latency is shown as the 'node distances' table in the output above.

**If the above command shows only a single node, then the system is not configured to take advantage of NUMA affinity.**

However, the `numactl` tool does not show the NUMA affinity of the GPUs in the system. In order to find this information, use `nvidia-smi topo -m`:

```
$ nvidia-smi topo -m

GPU0    GPU1    GPU2    GPU3    CPU    Affinity        NUMA Affinity
GPU0     X      SYS     SYS     SYS    48-63,112-127   3
GPU1    SYS      X      SYS     SYS    32-47,96-111    2
GPU2    SYS     SYS      X      SYS    16-31,80-95     1
GPU3    SYS     SYS     SYS      X     0-15,64-79      0

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
The above sample output suggests that GPU3 is in NUMA node 0. You can tell this either by looking at the 'NUMA Affinity' column, or matching the CPU core IDs from the 'Affinity' column with that from `numactl --hardware`. Therefore, performance would be improved if CPUs 0-15 and 65-79 were mapped to talk to GPU 3, using the memory address space of NUMA node 0.

In order to exploit the above NUMA affinity information, it is recommended to configure the harness accordingly. In v2.0, this step has been automated, as during startup, NVIDIA's MLPerf Framework will attempt to do the above steps to compute the NUMA configuration. From **inside the MLPerf Inference container**, launch a Python console and run the following commands:

```
>>> from code.common.systems.system_list import DETECTED_SYSTEM
>>> print(DETECTED_SYSTEM.numa_conf)
3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127
```
**On NVIDIA DGX Stations, this method will fail**, since the Mellanox NICs interfere with the steps above. However, this is fine because we **do not use NUMA on DGX Station machines**, since NUMA is meant to improve memory transfer between host and GPU. On DGX Stations, we use start_from_device, so this does not apply.

It is still **recommended** to sanity check and go through the **manual steps** below. If any mismatch is found between the manual steps and automated steps, **please report the bug to NVIDIA**. The **manual** steps to do this differ based on the type of harness:

- Triton harnesses, LWIS, and other "custom" harnesses (i.e. `harness_dlrm`, `harness_bert`) require users to specify the `numa_config` key in the system's `BenchmarkConfiguration` in the appropriate Python file. The value of this key is specified as a string with the following convention:

```
[NUMA node 0 config]&[NUMA node 1 config]&[NUMA node 2 config]&...

where [NUMA node n config] is a string in the format: [GPU ID(s)]:[CPU ID(s)]
where [___ ID(s)] can be a single digit, comma-separated digits, or numeric ranges using dashes.
```
For the examples above, the `numa_config` value would be:

```
numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"
```
This value maps:

    - NUMA node 0 to GPU 3, with CPUs 0-15 and 64-79

    - NUMA node 1 to GPU 2, with CPUs 16-31 and 80-95
    - NUMA node 2 to GPU 1, with CPUs 32-47 and 96-111
    - NUMA node 3 to GPU 0, with CPUs 48-63 and 112-127
- The Multi-MIG Triton harness finds NUMA aware settings automatically.

### System configuration tips

For systems with passively cooled GPUs, and especially NVIDIA T4-based systems, the cooling system in hardware plays an important role in performance. To check for thermal throttling, run `nvidia-smi dmon -s pc` to monitor the GPU temperature while the harness is running. Ideally, the GPU temperature should saturate or stabilize to a reasonable temperature, such as 65C. If the temperature is erratic or spiking, or if the GPU clock frequencies are unstable, you may need to improve your system's cooling solution.

On Jetson platforms, use `tegrastats` instead of `nvidia-smi` to monitor GPU temperature.

In addition, on NVIDIA T4-based systems, lock the GPU clocks at max frequency with

```
sudo nvidia-smi -lgc 1590,1590
```
For systems when running in Server scenario, please make sure that the Transparent Huge Page system setting is set to 'always'.

### Engine build failures

Sometimes, when simply copy-pasting configuration blocks, you can run into hard crashes in the `generate_engines` (engine build) step. Here are some example error messages:

```
$ make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --test_run"
...
[2021-06-23 02:05:06,760 builder.py:151 INFO] Building ./build/engines/GA104x1/resnet50/Offline/resnet50-Offline-gpu-b1024-int8.default.plan
[TensorRT] INFO: [MemUsageSnapshot] Builder begin: CPU 1046 MiB, GPU 527 MiB
[TensorRT] INFO: Reading Calibration Cache for calibrator: EntropyCalibration2
[TensorRT] INFO: Generated calibration scales using calibration cache. Make sure that calibration cache has latest scales.
[TensorRT] INFO: To regenerate calibration cache, please delete the existing one. TensorRT will generate a new calibration cache.
[TensorRT] INFO: [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +747, GPU +316, now: CPU 1793, GPU 843 (MiB)
[TensorRT] INFO: [MemUsageChange] Init cuDNN: CPU +619, GPU +268, now: CPU 2412, GPU 1111 (MiB)
[TensorRT] WARNING: Detected invalid timing cache, setup a local cache instead
[TensorRT] INFO: [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +0, now: CPU 2602, GPU 1173 (MiB)
[TensorRT] ERROR: 1: [convolutionRunner.cpp::executeConv::454] Error Code 1: Cudnn (CUDNN_STATUS_EXECUTION_FAILED)
Process Process-1:
Traceback (most recent call last):
...

$ make generate_engines RUN_ARGS="--benchmarks=dlrm --scenarios=offline --test_run"
...
[2021-06-23 01:57:38,832 builder.py:151 INFO] Building ./build/engines/GA104x1/dlrm/Offline/dlrm-Offline-gpu-b204000-int8.default.plan
[TensorRT] INFO: [MemUsageSnapshot] Builder begin: CPU 615 MiB, GPU 535 MiB
[TensorRT] INFO: Reading Calibration Cache for calibrator: EntropyCalibration2
[TensorRT] INFO: Generated calibration scales using calibration cache. Make sure that calibration cache has latest scales.
[TensorRT] INFO: To regenerate calibration cache, please delete the existing one. TensorRT will generate a new calibration cache.
[TensorRT] INFO: [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +748, GPU +318, now: CPU 1365, GPU 853 (MiB)
[TensorRT] INFO: [MemUsageChange] Init cuDNN: CPU +618, GPU +266, now: CPU 1983, GPU 1119 (MiB)
[TensorRT] WARNING: Detected invalid timing cache, setup a local cache instead
[TensorRT] INFO: Detected 2 inputs and 1 output network tensors.
[TensorRT] INFO: Total Host Persistent Memory: 3968
[TensorRT] INFO: Total Device Persistent Memory: 8704
[TensorRT] INFO: Total Scratch Memory: 0
[TensorRT] INFO: [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 4 MiB
[TensorRT] INFO: [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2881, GPU 1533 (MiB)
[TensorRT] INFO: [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 2881, GPU 1541 (MiB)
[TensorRT] INTERNAL ERROR: Assertion failed: out of memory
/work/code/plugin/DLRMInteractionsPlugin/src/dlrmInteractionsPlugin.cpp:224
Aborting...
Traceback (most recent call last):
...
```
While there is no sure-fire way to determine the exact error, you can go through the following possibilities to try and resolve the problem:

1. Out-of-memory error caused by a batch size that is too large
    1. In the above examples, if a CUDNN kernel fails to execute, or a plugin throws an OOM error, it is possible that the provided batch size is too large. This can sometimes be diagnosed if the SingleStream engine builds correctly, if the system is an Edge submission. To check for an fix this problem, set `gpu_batch_size` to 1. If the engine builds successfully, you can iteratively increase the batch size until it fails again. We recommend going by powers of 2 first to speed up this process.
2. Try disabling CUDA graphs.

### Using expected runtime to recognize issues

The [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) requires each submission to meet certain requirements to become a valid submission. One of these is the requirements is the runtime of the benchmark test. Below, we summarize some mathematical formulas to determine approximate benchmark runtimes.

**Offline:**

Default test runtime:

```
max((1.1 * min_duration * offline_expected_qps / actual_qps), (min_query_count / actual_qps))
```
Where:

- `min_duration`: 600 seconds by default, 60 seconds with `--test_run`
- `offline_expected_qps`: set by the benchmark's BenchmarkConfiguration
- `min_query_count`: 24576 by default

The typical runtime of Offline scenario should be around 660 seconds with default settings, 66 seconds with `--test_run` enabled. If the system has a very low QPS, the runtime will be much longer than expected.

**Server:**

Default test runtime:

```
(min_duration * server_target_qps / actual_qps)
```
Recommended test runtime:

```
max((min_duration * server_target_qps / actual_qps), (min_query_count / actual_qps))
```
Where:

- `min_duration`: 600 seconds by default, 60 seconds with `--test_run`
- `server_target_qps`: set by the benchmark's BenchmarkConfiguration
- `min_query_count`: 270336 by default

The typical runtime for Server scenario is about 600 seconds (60 seconds with `--test_run`) with `server_target_qps` is equal to or lower than the actual QPS. Otherwise, the runtime will be longer since queries will form a backlog as the SUT cannot process and return the queries in time.

Depending on the performance and latency behavior of the system, the runtime can keep increasing in "Default" mode because the Early Stopping mechanism decides that the current number of samples do not have enough statistical guarantees.

**Early stopping**: under the new rules, *min_query_count* is no longer a requirement for runtime if it becomes prohibitively long. *Early Stopping* allows for systems to process a smaller number of queries during their runs than previously allowed. After we have run for *min_duration*, the system can ask whether the overall processed number of queries already provide statistical guarantees of the reported performance number. If that is not the case, the system will suggest an extended number of queries to run. The system should stop at any case if the current processed number of queries reaches *min_query_count* as is unlikely that running any further would improve the chances of having a successful run (or actual convergence at all, meaning unbounded runtime).



**SingleStream:**

Default test runtime:

```
(min_duration / single_stream_expected_latency_ns * actual_latency)
```
Recommended test runtime:

```
max((min_duration / single_stream_expected_latency_ns * actual_latency), (min_query_count * actual_latency))
```
Where:

- `min_duration`: 600 seconds by default
- `single_stream_expected_latency_ns`: set by the benchmark's BenchmarkConfiguration
- `min_query_count`: 1024 by default

The typical runtime for SingleStream scenario should be about 600 seconds (60 seconds with `--test_run`), unless on a system with a very long latency per sample. In this case, runtime will be much longer.

**Early stopping**: under the new rules, *min_query_count* is no longer a requirement for runtime if it becomes prohibitively long. *Early Stopping* allows for systems to process a smaller number of queries during their runs than previously allowed. After we have run for *min_duration*, the system will report a conservative estimate of latency by adjusting the actual percentile of the reported result based on the actual number of queries processed. It is still recommended to run an increased number of samples (up to *min_query_count*) to improve the final reported result.

**MultiStream:**

Default test runtime:

```
(min_duration / multi_stream_expected_latency_ns * actual_latency)
```
Recommended test runtime:

```
max((min_duration / multi_stream_expected_latency_ns * actual_latency), (min_query_count * actual_latency))
```
Where:

- `min_duration`: 600 seconds by default
- `multi_stream_expected_latency_ns`: set by the benchmark's BenchmarkConfiguration
- `min_query_count`: 270336 by default

The typical runtime for MultiStream scenario should be about 600 seconds (60 seconds with `--test_run`), unless on a system with a very long latency per sample. In this case, runtime will be much longer.

**Early stopping**: under the new rules, *min_query_count* is no longer a requirement for runtime if it becomes prohibitively long. *Early Stopping* allows for systems to process a smaller number of queries during their runs than previously allowed. After we have run for *min_duration*, the system will report a conservative estimate of performance (in queries per second) by adjusting the actual percentile of the reported result based on the actual number of queries processed. It is still recommended to run an increased number of samples (up to *min_query_count*) to improve the final reported result.

### Fixing INVALID results

An INVALID result occurs when the harness finishes running successfully, but does not fulfill all of the runtime requirements to be considered valid for an official submission.

**Offline:**

The most common reason for INVALID results in Offline scenario is the actual QPS of the run was *too high* compared to the offline expected QPS. The reason this is reported as INVALID is that this behavior indicates that LoadGen was unable to generate a large enough load to saturate the accelerator. This can have multiple implications, such as not being able to cause the accelerator to reach thermal equilibrium.

For example, if the expected QPS for LoadGen was set to 1000, but the actual QPS of the run ended up to be 1500, it is not accurate to say the true QPS of the system is 1500, as with a smaller load, it is possible the accelerator was able to achieve a 1500 qps throughput at a lower temperature. It is completely possible that when the expected QPS for LoadGen is set to 1500, the accelerator can achieve thermal equilibrium and result in an actual QPS of something like 1300 because of higher temps.

To fix this, simply increase `offline_expected_qps` until the 'max query latency' reported by LoadGen reaches 660 (or 66 seconds depending on the `min_duration` setting), which is when `offline_expected_qps` matches the actual QPS.

**Server:**

The most common reason for INVALID results in Server scenario is that the actual QPS of the system is lower than the specified 'expected Server target QPS'. If this occurs, it will exhibit in the 99th percentile latency (described at the end of the LoadGen output) exceeding the [Server scenario latency target for the benchmark](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#constraints-for-the-closed-division), which is the criteria to be a valid submission.

To fix this, first try reducing the `server_target_qps` until the 99th percentile latency falls below the [Server latency targets](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#constraints-for-the-closed-division). If this does not resolve the 99th percentile latency issues, try reducing the `gpu_batch_size` (requires rebuilding the engines each time) and/or `gpu_inference_streams`.

**SingleStream:**

The most common reason for INVALID results in SingleStream scenario is that the actual latency of the system per sample is much lower than the specified SingleStream expected latency. Therefore, simply lower `single_stream_expected_latency_ns` to match the actual 90th percentile latency reported by LoadGen.

**MultiStream:**

The most common reason for INVALID results in MultiStream scenario is that the actual latency of the system per sample is much lower than the specified MultiStream expected latency. Therefore, simply lower `multi_stream_expected_latency_ns` to match the actual 99th percentile 'query' latency reported by LoadGen.

### Tuning parameters for better performance

It is possible to get better performance by tuning various parameters in the config files. To perform such tuning experiments, you can use our parameter-grid-search tool at `closed/NVIDIA/scripts/autotune`. This script has its own README.md to describes its instructions and usage. Note that there is also a subdirectory within the script called `mlperf1.0` that includes the grid search parameter files we used to tune in MLPerf Inference v1.0. Feel free to use those files as a template.

Note that there are benchmark-specific and platform-specific parameters. For example, on Jetson systems, the CV networks (ResNet50, SSDResNet34, SSDMobileNet) also have DLA parameters, such as `dla_batch_size`, that can also be tuned. See `closed/NVIDIA/common/fields.py` to understand the more specific parameters.

**Offline:**

In the Offline scenario, default settings should provide near-optimal performance in most cases. You can try modifying the following parameters:

- `gpu_batch_size`
- `gpu_copy_streams`
- `gpu_inference_streams`

**Server:**

In the Server scenario, the goal is to increase `server_target_qps` to its maximal value possible that can still satisfy the [MLPerf Inference server latency requirements](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#constraints-for-the-closed-division).

This can be done in an iterative process:

1. Increase `server_target_qps` to the maximum value that still meets the latency constraints with the current settings.
2. Sweep across many variations of the existing settings. This can be done with the parameter-grid-search tool by holding server_target_qps constant once you find it.
3. Replace the current settings with those providing the lowest latency at the target for the current server_target_qps.
4. Repeat from step 1 until 'server_target_qps' no longer needs to be tuned

**SingleStream:**

In the SingleStream scenario, there are not many values to tune, since the scenario is so simple. All you need to do is follow the steps from previous sections until it reports a VALID result, mainly with the following parameter:

- `single_stream_expected_latency_ns`

**MultiStream:**

MultiStream is very similar to to SingleStream scenario, except that now each query is composed of 8 samples (or as in multi_stream_samples_per_query). As in SingleStream, it is expected to process query by query, i.e. no other query will be requested until the previously requested query is processed and returned.

Performance tuning is largely upon selecting best performing batch size. The largest batch size is at most 8 samples, and how to group the samples for a batch needs to be considered to finish exactly 8 samples.

First tune for the best performance within 8 samples budget, and then follow the steps from previous sections until it reports a VALID result, with the following parameters:

- `gpu_batch_size`
- `multi_stream_expected_latency_ns`

