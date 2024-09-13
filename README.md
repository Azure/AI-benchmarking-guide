# Comprehensive Benchmarking Guide

Usage: `python3 runner.py`

If your machine does not have pip packages, nccl, and docker, run `bash install_requirements.sh` to install them prior to running `runner.py`. 

If you only want to run a specific test, see  `runner.py` and comment out the tests you are not interested in running. 

Test results will be stored in the `Outputs` directory.

See `config.json` for specific settings for the benchmarks

## Tests Included: 

### 1. CublasLt GEMM
The CuBLASLt General Matrix-to-matrix Multiply (GEMM) is a performance evaluation test for the CUDA Basic Linear Algebra Subroutines (CuBLAS) library for matrix and vector operations that leverages the parallel processing capabilities of GPUs. The benchmark is designed to assess the speed of matrix-to-matrix multiplication, which is the fundamental operation in AI applications, by measuring for varying matrix sizes (m, n, and k). The results shown below are with random initialization (best representation of real-life workloads) and datatype FP8.

In the guide, we run CuBLASLt on various matrix sizes. See the `run_model_sizes` function in `GEMMCublasLt.py`. 

For Power & Clock Frequency analysis, see `run_nvml`. It consists of M=N=K=8192 CuBLASLt GEMM ran repeatedly over 120 seconds, precision FP8. The power draw, clock frequency, and GPU temperature are measured and charted over this interval. 

For the sweeps over various values of m, n, and k, see `run_shmoo`. This takes up the most time, so feel free to comment out the call in `runner.py` if you are not interested in running this. It generates plots for m,n,k values, allowing you to see the performance over a range of matrix sizes.

### 2. Flash Attention
FlashAttention is an algorithm to speed up attention and reduce the memory footprint for Natural Language Models—without any approximation. It is meant to speed up training and inference by reordering the attention computation and leveraging classical techniques (tiling, recomputation) to reduce memory usage from quadratic to linear in sequence length. 

### 3. NCCL Bandwidth

The NCCL bandwidth test is a benchmark provided by NVIDIA's NCCL (NVIDIA Collective Communications Library) library. NCCL is a high-performance library, designed to accelerate interGPU communication, that optimizes communication between multiple GPUs within a single node or across multiple nodes in a multi-GPU system. 
The performance measured is the data transfer bandwidth between GPUs using various communication patterns, such as point-to-point (pairwise) communication or collective communication (communication between multiple GPUs). 

### 4. HBM Bandwidth
High Bandwidth Memory (HBM) is designed to provide a significant boost in memory bandwidth for GPUs by handling vast amounts of data through vertical stacking of multiple layers of memory chips, connected by through-silicon vias. 

### 5. NV Bandwidth
The NV Bandwidth benchmark measures the bandwidth achieved while transferring packets CPU-to-GPU and GPU-to-CPU over PCIe, and GPU-to-GPU over NVLink. 

### 6. LLM Benchmarks
This section contains benchmarking tests for various Inference models. The tests are performed in a TensorRT-LLM environment.

You need huggingface credentials to download all the model weigths. Visit huggingface.co to create an account and obtain access to the models.

If you already have a huggingface account, add your credentials (hf username and password) under `credentials` in `config.json`. Please remember to remove these once you are done.

 The `models` field in `config.json` contains all the models that can be benchmarked. To run benchmark for a specific model, set `use_model: true`. They are all set to `false` by default.

 Results of all the runs will be pasted in `Outputs/LLMBenchmark_[machine name].csv`
