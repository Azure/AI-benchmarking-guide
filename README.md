# Azure AI Benchmarking Guide

Performance benchmarks for Azure GPU SKUs — microbenchmarks, workload tests, and LLM inference/training evaluations to help users identify bottlenecks and optimize cost/performance.

### Supported SKUs

| NVIDIA | AMD |
|--------|-----|
| ND A100 v4 | ND MI300X v5 |
| ND H100 v5 | |
| ND H200 v5 | |
| ND GB200 v6 | |
| ND GB300 v6 | |

---

## Tests Included — NVIDIA

### Microbenchmarks

1. **[CuBLASLt GEMM](Benchmarks/NVIDIA/GEMMCublasLt.py)** — Measures matrix multiplication (GEMM) throughput using CuBLASLt across varying matrix sizes (m, n, k) with random initialization. Supports `fp8e4m3`, `fp4e2m1`, and `fp16` datatypes (configurable in `config.json`).

2. **[NCCL Bandwidth](Benchmarks/NVIDIA/NCCLBandwidth.py)** — Measures inter-GPU data transfer bandwidth using NCCL across point-to-point and collective communication patterns (single-node and multi-node).
   > 📎 [Wiki for Debugging Multi-Node NCCL Performance](https://dev.azure.com/msazure/AzureWiki/_wiki/wikis/AzureWiki.wiki/781566/Debugging-NCCL-Performance-Issues) *(Azure Internal Only)*

3. **[HBM Bandwidth](Benchmarks/NVIDIA/HBMBandwidth.py)** — Measures GPU High Bandwidth Memory throughput (vertically stacked memory chips connected via through-silicon vias).

4. **[NV Bandwidth](Benchmarks/NVIDIA/NVBandwidth.py)** — Measures CPU↔GPU bandwidth over PCIe and GPU↔GPU bandwidth over NVLink.

5. **[Flash Attention](Benchmarks/NVIDIA/FlashAttention.py)** — Benchmarks [FlashAttention](https://github.com/Dao-AILab/flash-attention), an IO-aware exact attention algorithm that uses tiling and recomputation to reduce memory usage from quadratic to linear in sequence length.

6. **[CPU STREAM](Benchmarks/NVIDIA/CPUStream.py)** — Measures CPU↔RAM memory bandwidth, critical for memory-intensive HPC workloads.

7. **[Multichase](Benchmarks/NVIDIA/Multichase.py)** — Measures random memory access (pointer-chasing) latency across the cache hierarchy, important for irregular access patterns (databases, graph processing).

### Workload Benchmarks

8. **[LLM Inference](Benchmarks/NVIDIA/LLMBenchmark.py)** — Benchmarks LLM inference throughput (tokens/sec) using TensorRT-LLM with Llama 3 (8B, 70B, 405B). Requires HuggingFace credentials and TensorRT container.

9. **[Llama 3 Pretraining](Benchmarks/NVIDIA/LLAMA3Run.py)** — Benchmarks single-node pretraining (Llama 3 3B, 8B) using NeMo, measured in time per step. Requires NeMo credentials to pull the container.

---

## Tests Included — AMD

### Microbenchmarks

1. **[HipBLASLt GEMM](Benchmarks/AMD/GEMMHipblasLt.py)** — AMD equivalent of CuBLASLt GEMM; measures matrix multiplication throughput using HipBLASLt across varying matrix sizes with random initialization (FP8).

2. **[RCCL Bandwidth](Benchmarks/AMD/RCCLBandwidth.py)** — AMD equivalent of NCCL; measures inter-GPU communication bandwidth using RCCL across point-to-point and collective patterns.

3. **[HBM Bandwidth](Benchmarks/AMD/HBMBandwidth.py)** — Measures GPU HBM throughput on MI300X.

4. **[TransferBench](Benchmarks/AMD/TransferBench.py)** — Measures CPU↔GPU transfer bandwidth.

5. **[Flash Attention](Benchmarks/AMD/FlashAttention.py)** — Benchmarks FlashAttention on ROCm, using tiling and recomputation to reduce memory usage from quadratic to linear.

### Workload Benchmarks

6. **[LLM Inference](Benchmarks/AMD/LLMBenchmark.py)** — Benchmarks LLM inference throughput (tokens/sec) using vLLM with Llama 3 (8B, 70B, 405B). Requires HuggingFace credentials.

---

# Benchmark Setup & Execution

### 1. Install Dependencies

```bash
# Recommended: use a virtual environment
python3 -m venv venv && source venv/bin/activate

# Install (auto-detects NVIDIA or AMD GPU)
./install-dependencies.sh
```

### 2. Storage Requirements

- **≥ 5 TB disk** recommended if running LLM benchmarks (model weights are large).
- Some AMD benchmarks use Docker containers (auto-created and killed). To avoid filling your boot disk, redirect Docker's data directory:

  ```bash
  # /etc/docker/daemon.json
  {
      "data-root": "/mnt/resource_nvme/docker"
  }
  ```

  > Verify the NVMe mount path on your machine. You may need to complete Docker [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).


### 3. LLM Benchmark Setup 

Pull and launch the TensorRT-LLM container (NVIDIA Platforms Only):


```bash
# Make sure you are in AI-benchmarking-guide directory
 
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14

docker run --rm -it --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:$PWD \
  -w $PWD \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14
```

Inside the container, set HuggingFace home and login:

```bash
export HF_HOME=$PWD
huggingface-cli login
```

> Get your token from [huggingface.co](https://huggingface.co/).

### 4. Run Benchmarks

**NVIDIA** — `python3 NVIDIA_runner.py [arg1] [arg2] ...`

| Argument | Test |
|----------|------|
| `gemm` | CuBLASLt GEMM |
| `nccl` | NCCL Bandwidth |
| `hbm` | HBM Bandwidth |
| `nv` | NV Bandwidth |
| `fa` | Flash Attention |
| `fio` | FIO (Storage I/O) |
| `cpustream` | CPU STREAM |
| `multichase` | Multichase |
| `llm` | LLM Inference |
| `llama_8b_pretrain` | Llama 3 8B Pretrain |
| `llama_3b_pretrain` | Llama 3 3B Pretrain |

**AMD** — `python3 AMD_runner.py [arg1] [arg2] ...`

| Argument | Test |
|----------|------|
| `gemm` | HipBLAS GEMM |
| `rccl` | RCCL Bandwidth |
| `hbm` | HBM Bandwidth |
| `transfer` | TransferBench |
| `fa` | Flash Attention |
| `fio` | FIO (Storage I/O) |
| `llm` | LLM Inference |

### Configuration

- **[`config.json`](config.json)** controls LLM benchmark settings (models, input/output sizes, tensor parallelism, precision, pretraining params).
- To benchmark a specific model, set `"use_model": true` in the [`models`](config.json#L7) field.
- NVIDIA models are marked `"type": "nvidia"`, AMD models are marked `"type": "amd"`.

### Output

- Results → `Outputs/` directory (markdown files).
- Logs → `Outputs/log.txt`.
  
---

## Azure Reference Results

Pre-validated results for each SKU are available in [`Azure_Results/`](Azure_Results/).
