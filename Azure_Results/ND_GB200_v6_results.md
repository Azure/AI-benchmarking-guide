# Azure ND GB200 v6 Benchmark Results

## System Specifications

| GPU           | NVIDIA GB200 185GB |
|---------------|-------------------|
| CPU           | ARM Neoverse-V2 |
| Ubuntu        |   24.04  |
| CUDA          |   12.8  |
| NVIDIA Driver | 570.133.20  |
| VBIOS         | 97.00.82.00.30 |
| NCCL          |    2.25.1 |
| PyTorch       |    2.6.0   |


## Microbenchmarks
### GEMM CuBLASLt (FP8)

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND GB200 v6 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     | 489.2                  |
| 2048        | 2048      | 2048     | 2243.6                |
| 4096        | 4096      | 4096     | 2834.6                |
| 8192        | 8192      | 8192     | 2876.6                 |
| 16384       | 16384     | 16384    | 2906.9                 |
| 32768       | 32768     | 32768    | 2825.6                 |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     | 1022.2                  |
| 6144        | 12288     | 12288    | 2895.6                |
| 802816      | 192       | 768      | 1459.5                  |

### GEMM CuBLASLt (FP4)

The results shown below are with random initialization (best representation of real-life workloads), FP4, and 10,000 warmup iterations.

| m           | n         | k        | ND GB200 v6 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     |   261.8                |
| 2048        | 2048      | 2048     |   1398.0              |
| 4096        | 4096      | 4096     |   4854.5              |
| 8192        | 8192      | 8192     |   7609.1              |
| 16384       | 16384     | 16384    |   5604.0               |
| 32768       | 32768     | 32768    |   5262.2              |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     | 548.3                  |
| 802816      | 192       | 768      | 2624.5                  |

### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|       | ND GB200 v6 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 147.6   |
| Flash Attention 2.0   | 373.9  |

### NV Bandwidth

|                       | ND GB200 v6 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 201                |
| Device to Host        | 193                |
| Device to Device read |  1530              |


### CPU STREAM

|       | ND GB200 v6 (GB/s) |
| ------| ------------------ |
| Copy  | 678                |
| Mul   | 687                |
| Add   | 768                |
| Triad | 763                |
| Dot   | 564                |


### NCCL Bandwidth

The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce (Ring algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND GB200 v6 (GB/s) |
| -------------------- | ----------------- |
| 1K          | 0.10            |
| 2K          | 0.19            |
| 4K          | 0.38            |
| 8K          | 0.76            |
| 16K         | 1.46            |
| 32K         | 2.88            |
| 65K         | 5.71            |
| 132K        | 11.29           |
| 256K        | 22.41           |
| 524K        | 44.07           |
| 1M          | 80.48           |
| 2M          | 110.43          |
| 4M          | 146.27          |
| 8M          | 253.46          |
| 16M         | 331.09          |
| 33M         | 422.62          |
| 67M         | 550.73          |
| 134M        | 574.19          |
| 268M        | 593.53          |
| 536M        | 618.74          |
| 1G          | 646.87          |
| 2G          | 666.63          |
| 4G          | 673.06          |
| 8G          | 679.61          |


## End-to-End Inference Workloads - TensorRT-LLM

### LLAMA 3.1 (8B)

Performance results for LLAMA 3.1 (8B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 1       | 128       | 128        | 41058                  |
| 1       | 128       | 1024       | 46703                  |
| 1       | 128       | 2048       | 39809                  |
| 1       | 500       | 2000       | 33836                  |
| 1       | 1024      | 1024       | 27944                  |
| 1       | 2048      | 2048       | 20341                  |

### LLAMA 3 (70B)

Performance results for LLAMA 3 (70B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 4       | 128       | 128        |  13715                 |
| 4       | 128       | 1024       |  21626                 |
| 4       | 128       | 2048       |  21374                 |
| 4       | 500       | 2000       |  18279                 |
| 4       | 1024      | 1024       |  12376                 |
| 4       | 2048      | 2048       |  10642                 |

## End-to-End Pretraining Workloads - Single Node
### LLAMA 3 (3B)

<img width="600" height="600" alt="LLAMA3_3b_Pretrain_Results-GB200" src="https://github.com/user-attachments/assets/1759087d-00bb-453a-8e0a-6be104c64c88" />

### LLAMA 3 (8B)
<img width="600" height="600" alt="LLAMA3_8b_Pretrain_Results-GB200" src="https://github.com/user-attachments/assets/326fa4d0-9ce7-473d-9507-a9b95069652a" />




