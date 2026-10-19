# Bandwidth specs of datacenter systems for MLPerf Inference 5.1

## Introduction
As per the rules of MLPerf Inference, submissions are required to prove that the systems used provide a certain level of ingress (network to loadgen trace) and egress (loadgen trace to network) bandwidth.  
Put simply, the throughput at which the accelerator accepts queries and generates responses should not exceed the maximum data bandwidth the system is capable of supporting.

Say the throughput of a benchmark run is X samples/second. It is necessary to document that, the system is capable of supporting at least X inputs/outputs from/to the network, per second.

In v5.0, NVIDIA submitted using the following datacenter systems:
- H200-SXM-141GBx8
- B200-SXM-180GBx8
- GB200-NVL72_GB200-186GB_aarch64x72

## Calculating the maximum permissible QPS
For each workload, the offline scenario uses the most bandwidth. We list the bandwidth used per model `used_bw` (in byte/sec) for a run in Offline scenario.  
The `used_bw` is a function of throughput of input/output samples `tput` (sample/second) and size of each input/output sample `bytes/sample`.

### Ingress bandwidth requirements

| Benchmark                | Formula                                                                                                               | Bandwidth used   (bytes)        | Values                                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| ResNet50                 | ```used_bw = tput x (C x H x W) x dtype_size = tput x (3 x 224 x 224) = tput x 150528```                              | ```used_bw = tput x 150528```   | ```H = W = 224; C =3; dtype_size = 1byte```                                                                          |
| BERT                     | ```used_bw = tput x (num_inputs x max_input_len x dtype_size)```                                                      | ```used_bw = tput x 4608```     | ```num_inputs = 3; max_input_len = 384; dtype_size = 4bytes```                                                       |
| DLRM                     | ```used_bw = tput x num_pairs_per_sample x ((num_numerical_inputs x dtype_0) + (num_categorical_inputs x dtype_1))``` | ```used_bw = tput x 35100```    | ```num_pairs_per_sample = 270; num_numerical_inputs = 13; num_categorical_inputs = 26; dtype_0 = 2B; dtype_1 = 4B``` |
| 3D U-Net                 | ```used_bw = tput x avg(C x D x H x W)```                                                                             | ```used_bw = tput x 32944795``` | ```avg = 32944795```                                                                                                 |
| Llama3.1-8B              | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 10240```     | ```max_input_len = 2560; dtype_size = 4B```                                                                         |
| Llama2-70B               | ```used_bw = tput x  max_input_len x dtype_size```                                                                    | ```used_bw = tput x 4096```     | ```max_input_len = 1024; dtype_size = 4B```                                                                          |
| MoE                      | ```used_bw = tput x  max_input_len x dtype_size```                                                                    | ```used_bw = tput x 8192```     | ```max_input_len = 2048; dtype_size = 4B```                                                                          |
| RetinaNet                | ```used_bw = tput x C x H x W x dtype_size```                                                                         | ```used_bw = tput x 1920000```  | ```H = W = 800; C = 3; dtype_size = 1B```                                                                            |
| SDXL                     | ```used_bw = num_inputs x max_prompt_len x dtype_size```                                                              | ```used_bw = tput x 308```      | ```num_inputs = 1; max_prompt_len = 77```                                                                            |
| Llama3-405B              | ```used_bw = tput x  max_input_len x dtype_size```                                                                    | ```used_bw = tput x 80000```    | ```max_input_len = 20000; dtype_size = 4B```                                                                         |
| R-GAT                    | negligible                                                                                                            | negligible                      | >0                                                                                                                   |
| DeepSeek-R1              | ```used_bw = tput x  max_input_len x dtype_size```                                                                    | ```used_bw = tput x12544```     | ```max_input_len = 3136; dtype_size = 4B```                                                                          |
| Whisper                  | ```used_bw = tput x audio_length x sample_rate x dtype_size```                                                        | ```used_bw = tput x 960000```   | ```max_audio_length = 30sec; sample_rate = 16000Hz; dtype_size=2B```                                                 |

### Egress bandwidth requirements

According to the [rules set out by MLCommons](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b2-egress-bandwidth), we only need to measure egress bandwidth for 3D U-Net, SDXL and DS-R1. 
| Benchmark | Formula                                                                                                               | Bandwidth used (bytes)          | Values                                                                                                               |
|-----------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 3D U-Net  | ```used_bw = tput x avg(C x D x H x W)```                                                                             | ```used_bw = tput x 32944795``` | ```avg = 32944795```                                                                                                 |
| SDXL      | ```used_bw = num_inputs x image_height x image_width x image_channel x dtype_size```                                  | ```used_bw = tput x 3145728```  | ```num_inputs = 1; image_height = image_width = 1024; image_channel = 3```                                           |
| DeepSeek-R1              | ```used_bw = tput x  max_output_len x dtype_size```                                                                    | ```used_bw = tput x 80000```     | ```max_input_len = 20000; dtype_size = 4B```                                                                          |


## Network bandwidth of NVIDIA's systems

### DGX-H100_H100-SXM-80GBx8 and H200-SXM-141GBx8 (DGX H100/H200)
The [DGX H100/H200 User guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) specifies the network card description as below.

2 x NVIDIA® ConnectX®-7 Dual Port Ethernet Cards. Each card provides the following speeds: 
- Ethernet (default): 400GbE, 200GbE, 100GbE, 50GbE, 40GbE, 25GbE, and 10GbE
- InfiniBand: Up to 400Gbps

Thus, each CX-7 card allows for at least 400Gbps via InfiniBand and 400GbE via ethernet, amounting to 800Gbps. Since each system has 2 CX-7 cards, the total bandwidth is at least 2 x 800Gbps = 1600Gbps = 200GB/s.

### B200-SXM-180GBx8 (DGX B200)
The [B200 User guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) specifies the network card description as below.
2 x NVIDIA® BlueField®-3 DPU Dual Port Cards. Each card provides the following speeds:
- Ethernet (1 port): 400GbE, 200GbE, 100GbE, 50GbE, 40GbE, 25GbE, and 10GbE
- InfiniBand (1 port): Up to 400Gbps

Thus, each BlueField-3 DPU card allows for at least 400Gbps via InfiniBand and 400GbE via ethernet, amounting to 800Gbps. Since each system has 2 BlueField-3 DPU cards, the total bandwidth is at least 2 x 800Gbps = 1600Gbps = 200GB/s.

### GB200-NVL72_GB200-186GB_aarch64x72 (GB200 NVL72)
GB200-NVL72_GB200-186GB_aarch64x72 has 18 compute nodes. Each compute node has 2 Grace CPUs.  
Each Grace CPU is connected to at least one BlueField-3 Smart NIC offering upto 400Gbps = 50GB/s. Thus, each node offers at least 100GB/s network bandwidth. With 18 nodes, GB200 NVL72 provides at least 1800GB/s aggregate network bandwidth.


## Max permissible QPS per system
Using the formulae in the previous section, we calculate for each system-benchmark pair the maximum permissible QPS by setting `system_bw = used_bw` and calculating for `tput`.
For workloads with constraint on both ingress and egress bandwidth, we take the max of the two. (for example, we take egress for SDXL)

PLEASE NOTE - The numbers are calculated below for NVIDIA's systems and are provided for reference only. Each systems configuration (and hence, bandwidth) may be different. It is imperative that each participant does such calculations individually for their own systems.

| System                       | Bandwidth of system (= bytes/sec)   | ResNet50      | BERT        | DLRM        | 3D U-Net | Llama2-70B  | Llama2-70B-Interactive | RetinaNet   | SDXL         | MoE         | Llama3-405B | Whisper      | Llama3.1-8B  | DeepSeek-R1  |
|------------------------------|-------------------------------------|---------------|-------------|-------------|----------|-------------|------------------------|-------------|--------------|-------------|-------------|--------------|---------------|--------------|
| H200-SXM-141GBx8             | 200GB/s (= 2 x 10^11)               | 1.32 x 10^6   | 4.34 x 10^7 | 4.52 x 10^7 | 6,070    | 4.8 x 10^7  | 4.8 x 10^7             | 104,166     | 63,578       | 2.4 x 10^7  | 2.5 x 10^6  | 2.08 x 10^5  | 1.95 x 10^7  | 2.5 x 10^6   |
| B200-SXM-180GBx8             | 200GB/s (= 2 x 10^11)               | 1.32 x 10^6   | 4.34 x 10^7 | 4.52 x 10^7 | 6,070    | 4.8 x 10^7  | 4.8 x 10^7             | 104,166     | 63,578       | 2.4 x 10^7  | 2.5 x 10^6  | 2.08 x 10^5  | 1.95 x 10^7  | 2.5 x 10^6   |
| GB200-NVL72_GB200-186GB_aarch64x72 | 1800GB/s (= 1.8 x 10^12)      | 1.19 x 10^7   | 3.91 x 10^8 | 4.07 x 10^8 | 54,630   | 4.32 x 10^8 | 4.32 x 10^8            | 937,494     | 572,202      | 2.16 x 10^8 | 2.25 x 10^7 | 1.875 x 10^6 | 1.76 x 10^8  | 2.25 x 10^7  |