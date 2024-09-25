# Azure ND H100 v5 Benchmark Results

## System Specifications

| GPU           | NVIDIA A100-SXM4-80GB |
|---------------|-------------------|
| CPU           | AMD EPYC 7V12 64-Core Processor |
| Ubuntu        |   22.04  |
| CUDA          |   12.5  |
| NVIDIA Driver | 550.90.07   |
| VBIOS         | 92.00.9E.00.01 |
| NCCL          |    2.18.5  |
| PyTorch       |    2.4.1   |


## Microbenchmarks
### GEMM CuBLASLt  

The results shown below are with random initialization (best representation of real-life workloads), FP16, and 10,000 warmup iterations.

| m           | n         | k        | ND A100 V4 (TFLOPS)    | 
| ----------- | --------- | -------- | ---------------------- |  
| 1024        | 1024      | 1024     | 119.4                   |  
| 2048        | 2048      | 2048     | 221.8               |  
| 4096        | 4096      | 4096     | 228.0                 |  
| 8192        | 8192      | 8192     | 253.7                |  
| 16384       | 16384     | 16384    | 256.5                |  
| \---------- | \-------- | \------- | \--------------------- |  
| 1024        | 2145      | 1024     | 132.8                   |  
| 6144        | 12288     | 12288    | 250.8                |  
| 802816      | 192       | 768      | 157.8                  |  

### HBM Bandwidth

|       | ND A100 V4 (TB/s) | 
| ----- | ----------------- |  
| Copy  | 1.61              |  
| Mul   | 1.61              |  
| Add   | 1.70              |  
| Triad | 1.70              |  
| Dot   | 1.75              |  


### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 64, a batch size of 2, and a sequence length of 8192.

|       | ND A100 V4 (TFLOPS) | 
| ----- | ----------------- |  
| Standard Attention(PyTorch)  | 46.0   |  
| Flash Attention 2.0   | 179.5  |

### NV Bandwidth

|                       | ND A100 V4 (GB/s) |  
| --------------------- | ----------------- |  
| Host to Device        | 26.28                |  
| Device to Host        | 6.27                |  
| Device to Device read | 423               |  


### FIO Tests

| Test             | Batch Size(Bytes) | ND A100 V4 (GB/s) |  
| ---------------- | ----------------- | ----------------- |  
| Sequential read  | 1M                | 9.3              |  
| Sequential read  | 512k              | 9.3              |  
| Random read      | 1k                | 0.4              |  
| Sequential read  | 1k                | 0.4             |  
| Sequential write | 1M                | 8.2              |  
| Sequential write | 512k              | 8.2              |  
| Random write     | 1k                | 0.35              |  
| Sequential write | 1k                | 0.39              |  


## NCCL Bandwidth

The values (in GB/s), in the table 6 and figure 5 below, are the bus bandwidth values obtained from the NCCL AllReduce (NVLS algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND A100 V4 (GB/s) |  
| -------------------- | ----------------- |  
| 1K                   | 0.04              |  
| 2K                   | 0.08              | 
| 4K                   | 0.17              |  
| 8K                   | 0.33              |  
| 16K                  | 0.67              |  
| 32K                  | 1.37             |  
| 65K                  | 2.70              |  
| 132K                 | 5.15              |  
| 256K                 | 10.11             |  
| 524K                 | 19.24             |  
| 1M                   | 36.22             |  
| 2M                   | 53.54             |  
| 4M                   | 71.43            |  
| 8M                   | 89.99            |  
| 16M                  | 109.22            |  
| 33M                  | 132.77            |  
| 67M                  | 166.63            |  
| 134M                 | 174.18            |  
| 268M                 | 213.87            |  
| 536M                 | 221.58            |  
| 1G                   | 228.10           |  
| 2G                   | 230.22            |  
| 4G                   | 230.48            |  
| 8G                   | 231.58            |  

## End-to-End Inference Workloads 

### Mistral (7B) 

Performance results for Mistral (7B) ran with input length 128 and output Length 128.

|                   | Batch Size | ND A100 V4 |  
| ----------------- | ---------- | ---------- |  
| Tokens per second | 64         |     |  

### LLAMA 3 (8B) 

Performance results for LLAMA 3 (8B) with input Length 128 and output length 128.

|                   | Batch Size | ND A100 V4 |  
| ----------------- | ---------- | ---------- |  
| Tokens per second | 64         |    |  

### LLAMA 3 (70B) 

Performance results for LLAMA 3 (70B) with world size 8, input length 128, and output length 128.

|                   | Batch Size | ND A100 V4 | 
| ----------------- | ---------- | ---------- | 
| Tokens per second | 16         |      |  
| Tokens per second | 32         |      | 
| Tokens per second | 64         |     |   

