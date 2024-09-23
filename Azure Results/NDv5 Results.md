# Azure NDv5 Benchmark Results

## System Specifications

| Specification | ND H200 v5                      | ND H100 v5                      |
| ------------- | ------------------------------- | ------------------------------- |
| GPU           | NVIDIA H200 141GB               | NVIDIA H100 80GB HBM3           |
| CPU           | Intel(R) Xeon(R) Platinum 8480C | Intel(R) Xeon(R) Platinum 8480C |
| Ubuntu        | 22.04                           | 22.04                           |
| CUDA          | 12.6                            | 12.6                            |
| NVIDIA Driver | 560.28.03                       | 560.35.03                       |
| VBIOS         | 96.00.9D.00.02                  | 96.00.84.00.03                  |
| NCCL          | 2.20.5                          | 2.18.5                          |
| Pytorch       | 2.4.0                           | 2.4.0                           |


## Microbenchmarks
### GEMM CuBLASLt  

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND H200 V5 (TFLOPS)    | ND H100 v5 (TFLOPS)     | Perf difference      |
| ----------- | --------- | -------- | ---------------------- | ----------------------- | -------------------- |
| 1024        | 1024      | 1024     | 290.8                  | 274.1                   | 5.8%                 |
| 2048        | 2048      | 2048     | 1013.4                 | 1011.3                  | 2.0%                 |
| 4096        | 4096      | 4096     | 1249.1                 | 1216.9                  | 2.7%                 |
| 8192        | 8192      | 8192     | 1300.6                 | 1290.1                  | \-0.8%               |
| 16384       | 16384     | 16384    | 1366.3                 | 1377.3                  | \-0.8%               |
| 32768       | 32768     | 32768    | 1349.5                 | 1382.2                  | \-2.4%               |
| \---------- | \-------- | \------- | \--------------------- | \---------------------- | \------------------- |
| 1024        | 2145      | 1024     | 407.9                  | 410.7                   | \-0.7%               |
| 6144        | 12288     | 12288    | 1318.0                 | 1351.4                  | \-2.4%               |
| 802816      | 192       | 768      | 717.5                  | 663.8                   | 8.1%                 |

### HBM Bandwidth

|       | ND H200 V5 (TB/s) | ND H100 v5 (TB/s) | % Speedup |
| ----- | ----------------- | ----------------- | --------- |
| Copy  | 4.01              | 2.90              | 38%       |
| Mul   | 4.00              | 2.90              | 38%       |
| Add   | 4.15              | 2.97              | 40%       |
| Triad | 4.15              | 2.97              | 40%       |
| Dot   | 4.48              | 3.18              | 41%       |

### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|                             | NVIDIA ND H200 V5 (TFLOPS) | ND H100 v5 (TFLOPS) |
| --------------------------- | ---------------------------- | ------------------- |
| Standard attention(PyTorch) | 161.5                        | 145.1               |
| Flash attention 2.0         | 329.3                        | 327.9               |

### NV Bandwidth

|                       | ND H200 V5 (GB/s) | ND H100 v5 (GB/s) |
| --------------------- | ----------------- | ----------------- |
| Host to Device        | 52                | 51                |
| Device to Host        | 52                | 52                |
| Device to Device read | 671               | 672               |

### FIO Tests

| Test             | Batch Size(Bytes) | ND H200 V5 (GB/s) | ND H100 v5 (GB/s) |
| ---------------- | ----------------- | ----------------- | ----------------- |
| Sequential read  | 1M                | 55.4              | 55.3              |
| Sequential read  | 512k              | 55.6              | 55.4              |
| Random read      | 1k                | 1.26              | 1.27              |
| Sequential read  | 1k                | 1.56              | 1.61              |
| Sequential write | 1M                | 33.8              | 33.9              |
| Sequential write | 512k              | 33.9              | 33.9              |
| Random write     | 1k                | 0.37              | 0.37              |
| Sequential write | 1k                | 0.39              | 0.39              |  


## NCCL Bandwidth

The values (in GB/s), in the table 6 and figure 5 below, are the bus bandwidth values obtained from the NCCL AllReduce (NVLS algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND H200 V5 (GB/s) | ND H100 v5 (GB/s) |
| -------------------- | ----------------- | ----------------- |
| 1K                   | 0.06              | 0.07              |
| 2K                   | 0.12              | 0.14              |
| 4K                   | 0.23              | 0.27              |
| 8K                   | 0.47              | 0.53              |
| 16K                  | 0.93              | 0.95              |
| 32K                  | 1.84              | 1.71              |
| 65K                  | 3.72              | 2.72              |
| 132K                 | 7.39              | 5.40              |
| 256K                 | 14.75             | 10.86             |
| 524K                 | 29.38             | 21.35             |
| 1M                   | 52.96             | 43.7              |
| 2M                   | 88.78             | 81.93             |
| 4M                   | 138.44            | 128.73            |
| 8M                   | 181.94            | 172.13            |
| 16M                  | 240.41            | 234.45            |
| 33M                  | 300.91            | 295.53            |
| 67M                  | 368.11            | 361.78            |
| 134M                 | 412.45            | 400.61            |
| 268M                 | 432.24            | 423.38            |
| 536M                 | 442.72            | 438.46            |
| 1G                   | 467.58            | 466.35            |
| 2G                   | 472.57            | 472.36            |
| 4G                   | 477.41            | 475.54            |
| 8G                   | 479.98            | 478.87            | 

## End-to-End Inference Workloads 

### Mistral (7B) 

Performance results for Mistral (7B) ran with input length 128 and output Length 8.

|                   | Batch Size | ND H200 V5 | ND H100 v5 | Perf difference |
| ----------------- | ---------- | ---------- | ---------- | --------------- |
| Tokens per second | 64         | 2822.37    | 2696.04    | 4.7%            | 

### LLAMA 3 (8B) 

Performance results for LLAMA 3 (8B) with input Length 128 and output length 8.

|                   | Batch Size | ND H200 V5 | ND H100 v5 | Perf difference |
| ----------------- | ---------- | ---------- | ---------- | --------------- |
| Tokens per second | 64         | 2757.54    | 2640.07    | 4.4%            |

### LLAMA 3 (70B) 

Performance results for LLAMA 3 (70B) with world size 8, input length 128, and output length 8.

|                   | Batch Size | ND H200 V5 | ND H100 v5 | Perf difference |
| ----------------- | ---------- | ---------- | ---------- | --------------- |
| Tokens per second | 16         | 647.78     | 624.57     | 3.7%            |
| Tokens per second | 32         | 861.77     | 825.02     | 4.4%            |
| Tokens per second | 64         | 1145.14    | 1077.84    | 6.3%            | 

### LLAMA 3 (405B) 

Performance results for LLAMA 3 (405B) with world size 8, input length 128, output length 8.

|                   | ND H200 V5 | ND H100 v5 | Perf difference |
| ----------------- | ---------- | ---------- | --------------- |
| Tokens per second | 333.47     | 251.5      | 32.7%           |
