# Azure AI Benchmarking Guide

Inefficient workload optimization can significantly increase operational costs for customers, making it essential to define clear performance benchmarks. This benchmarking guide establishes performance standards across a series of microbenchmarks, tests, and language models. These results are designed to help Azure users maximize efficiency, identify bottlenecks, and fine-tune resource allocation on Azure. By providing detailed performance insights, this guide ensures users can optimize workloads for both cost and performance, improving overall cloud infrastructure utilization. The guide currently supports the following SKUs:

### NVIDIA SKUs
- ND A100 v4
- ND H100 v5
- ND H200 v5
- ND GB200 v6
- ND GB300 v6

### AMD SKUs
- ND MI300X v5

## Tests Included - NVIDIA

### 1. Microbenchmark - CublasLt GEMM
[CuBLASLt General Matrix-to-matrix Multiply](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/GEMMCublasLt.py) (GEMM) is a performance evaluation test for the CUDA Basic Linear Algebra Subroutines (CuBLAS) library for matrix and vector operations that leverages the parallel processing capabilities of GPUs. The benchmark is designed to assess the speed of matrix-to-matrix multiplication, which is the fundamental operation in AI applications, by measuring for varying matrix sizes (m, n, and k). The results shown below are with random initialization (best representation of real-life workloads) and datatype FP8.

The guide supports 3 datatypes: `fp8e4m3`, `fp4e2m1` and `fp16`. Change these in `config.json`.

### 2. Microbenchmark - NCCL Bandwidth

The [NCCL bandwidth test](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/NCCLBandwidth.py) is a benchmark provided by NVIDIA's NCCL (NVIDIA Collective Communications Library) library. NCCL is a high-performance library, designed to accelerate interGPU communication, that optimizes communication between multiple GPUs within a single node or across multiple nodes in a multi-GPU system.
The performance measured is the data transfer bandwidth between GPUs using various communication patterns, such as point-to-point (pairwise) communication or collective communication (communication between multiple GPUs).

#### Extra:
[Wiki for Debugging Multi-Node NCCL Performance](https://dev.azure.com/msazure/AzureWiki/_wiki/wikis/AzureWiki.wiki/781566/Debugging-NCCL-Performance-Issues?anchor=recommended-command-lines-for-ndv6-infiniband-skus-(assuming-hpcx-or-openmpi-for-mpi)%3A) - Azure Internal Only 

### 3. Microbenchmark - HBM Bandwidth
[High Bandwidth Memory](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/HBMBandwidth.py) (HBM) is designed to provide a significant boost in memory bandwidth for GPUs by handling vast amounts of data through vertical stacking of multiple layers of memory chips, connected by through-silicon vias.

### 4. Microbenchmark - NV Bandwidth
The [NV Bandwidth](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/NVBandwidth.py) benchmark measures the bandwidth achieved while transferring packets CPU-to-GPU and GPU-to-CPU over PCIe, and GPU-to-GPU over NVLink.

### 5. Microbenchmark - Flash Attention
[FlashAttention](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/FlashAttention.py) is an algorithm to speed up attention and reduce the memory footprint for Natural Language Models—without any approximation. It is meant to speed up training and inference by reordering the attention computation and leveraging classical techniques (tiling, recomputation) to reduce memory usage from quadratic to linear in sequence length.

### 6. CPU STREAM Benchmark 
The [CPU STREAM](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/CPUStream.py) benchmark measures memory bandwidth performance rather than raw CPU speed. It evaluates how efficiently a system can move data between the CPU and RAM, which is crucial for memory-intensive applications like scientific computing and HPC.

### 7. Multichase Benchmark 
The [Multichase](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/Multichase.py) benchmark is a memory latency benchmark designed to measure pointer-chasing latency in a system. Unlike traditional memory benchmarks like STREAM, which focus on memory bandwidth, Multichase is used to evaluate random memory access latency, which is crucial for workloads that rely on irregular memory access patterns, such as databases and graph processing.

### 8. End-to-end Inference Workloads
To assess how different system components (as tested by the microbenchmarks) affect overall performance, we suggetsing running some [end-to-end workloads](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/LLMBenchmark.py). The models we used for benchmarking are the current industry standards across various sizes: LLAMA 3 (8B, 70B, and 405B). The performance of the model inferencing (throughput) is measured in tokens per second, accounting for both processing input tokens and generating output tokens. The workloads run in a TensorRT-LLM environment. Users need huggingface credentials to download all the model weigths.

### 9. End-to-end Pretraining Workloads
To assess the overall performance of GB200 and H200, we suggest running some [end-to-end pretrain workloads](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/NVIDIA/LLAMA3Run.py). The models we used for benchmarking are the current industry standards across various sizes: LLAMA 3 (3B, 8B). The performance of the model is measured in pretraining time per step. The workloads run in a Docker environment. Users need NeMo credentials to pull the container.

## Tests Included - AMD

### 1. Microbenchmark - hipBLAS GEMM
The [hipBLASLt General Matrix-to-matrix Multiply](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/GEMMHipblasLt.py) (GEMM) is a performance evaluation test for the hip Basic Linear Algebra Subroutines (hipBLAS) library for matrix and vector operations that leverages the parallel processing capabilities of GPUs. The benchmark is designed to assess the speed of matrix-to-matrix multiplication, which is the fundamental operation in AI applications, by measuring for varying matrix sizes (m, n, and k). The results shown below are with random initialization (best representation of real-life workloads) and datatype FP8.


### 2. Microbenchmark - RCCL Bandwidth
The [RCCL bandwidth test](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/RCCLBandwidth.py) is a benchmark provided by ROCm RCCL (ROCm Collective Communications Library) library. RCCL is a high-performance library, designed to accelerate interGPU communication, that optimizes communication between multiple GPUs within a single node or across multiple nodes in a multi-GPU system.
The performance measured is the data transfer bandwidth between GPUs using various communication patterns, such as point-to-point (pairwise) communication or collective communication (communication between multiple GPUs).

### 3. Microbenchmark - HBM Bandwidth
[High Bandwidth Memory](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/HBMBandwidth.py) (HBM) is designed to provide a significant boost in memory bandwidth for GPUs by handling vast amounts of data through vertical stacking of multiple layers of memory chips, connected by through-silicon vias.

### 4. Microbenchmark - TransferBench
The [NV Bandwidth](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/TransferBench.py) benchmark measures the bandwidth achieved while transferring packets CPU-to-GPU and GPU-to-CPU.

### 5. Microbenchmark - Flash Attention
[FlashAttention](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/FlashAttention.py) is an algorithm to speed up attention and reduce the memory footprint for Natural Language Models—without any approximation. It is meant to speed up training and inference by reordering the attention computation and leveraging classical techniques (tiling, recomputation) to reduce memory usage from quadratic to linear in sequence length.

### 6. End-to-end Inference Workloads
To assess how different system components (as tested by the microbenchmarks) affect overall performance, we suggetsing running some [end-to-end workloads](https://github.com/Azure/AI-benchmarking-guide/blob/main/Benchmarks/AMD/LLMBenchmark.py). The models we used for benchmarking are the current industry standards across various sizes: LLAMA 3 (8B, 70B, and 405B). The performance of the model inferencing (throughput) is measured in tokens per second, accounting for both processing input tokens and generating output tokens. The workloads run in a vLLM environment. Users need huggingface credentials to download all the model weigths. 


# HOW TO RUN THE BENCHMARKS

We highly recommend running the benchmarks inside a virtual env to avoid clashing with other pip dependencies. Start a virtual env and activate it:

```
python3 -m venv venv && source venv/bin/activate
```

Installation of benchmark dependencies requires multiple steps.
A convenience script `install-dependencies.sh` is provided to simplify installation.

```bash
./install-dependencies.sh 
```

If you wish to run LLM benchmarks, make sure to correctly set the huggingface home directory. This is where the model weights will be downloaded:

```
export HF_HOME=$PWD
```
Then login with your huggingface token (obtained from [huggingface.co](https://huggingface.co/))

```
huggingface-cli login
```

### NVIDIA

Usage: `python3 NVIDIA_runner.py [arg]`\
   or: `python3 NVIDIA_runner.py [arg1] [arg2]` ... to run more than one test e.g `python3 NVIDIA_runner.py hbm nccl`\
Arguments are as follows, and are case insensitive:\
CuBLASLt GEMM:   `gemm`\
NCCL Bandwidth:  `nccl`\
HBMBandwidth:    `hbm`\
NV Bandwidth:   `nv`\
Flash Attention: `fa`\
FIO Tests:   `fio`\
CPU Stream: `cpustream`\
Multichase:  `multichase`\
LLM Inference Workloads: `llm`\
LLAMA3 8B Pretrain Workload: `llama_8b_pretrain` \
LLAMA3 3B Pretrain Workload: `llama_3b_pretrain`

### AMD
Usage: `python3 AMD_runner.py [arg]`\
   or: `python3 AMD_runner.py [arg1] [arg2]` ... to run more than one test e.g `python3 AMD_runner.py hbm nccl`\
Arguments are as follows, and are case insensitive:\
HipBLAS GEMM:  `gemm`\
RCCL Bandwidth: `rccl`\
HBMBandwidth:   `hbm`\
TransferBench:   `transfer`\
Flash Attention: `fa`\
FIO Tests:   `fio`\
LLM Inference Workloads: `llm`

### Extras
- Test results will be stored in a markdown file in the `Outputs` directory.
- The console output and errors are logged in `Outputs/log.txt.`
- The file [`config.json`](https://github.com/Azure/AI-benchmarking-guide/blob/main/config.json) contains the specific settings for LLM benchmarks.
- The [`models`](https://github.com/Azure/AI-benchmarking-guide/blob/77a67867be39c0418cf958e9485454a3c4d7415b/config.json#L7) field in `config.json` contains all the inference models that can be benchmarked. To run benchmark for a specific model, set `use_model: true`. They are all set to `false` by default.
- All the AMD models in `config.json` are marked with `"type": "amd"`
- All the NVIDIA models in `config.json` are marked with `"type": "nvidia"`

You can find results of these benchmarks ran on various virtual machines in the [`Azure_Results`](https://github.com/Azure/AI-benchmarking-guide/tree/main/Azure_Results) directory.

### Storage
- We recommend cloning this benchmark repository onto a disk with at least 5TB if you plan on running LLM Benchmarks, because the model weights are massive. 
- Some of the AMD benchmarks are ran in docker containers, which are automatically created and killed when the tests are ran. To make sure that these docker containers don't fill up your storage space, change the default location that docker stores its files. Do this by:

  `vim /etc/docker/daemon.json`

  add a `data-root` field to `daemon.json` and make sure it points to a directory on the mounted NVMe disk, or somewhere with sufficient storage space (at least 1TB). for example:

  ```
  {
       "data-root":"/mnt/resource_nvme/docker", 
  }
  ```
  - Note that this won't be the path to the NVMe disk on all machines, so check to make sure.
  - You may need to perform the docker [`post-installation steps`](https://docs.docker.com/engine/install/linux-postinstall/) to complete the docker reconfiguration.
