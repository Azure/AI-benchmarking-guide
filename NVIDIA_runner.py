import os
import sys
import subprocess
import torch

from Benchmarks.NVIDIA import GEMMCublasLt as gemm
from Benchmarks.NVIDIA import HBMBandwidth as HBM
from Benchmarks.NVIDIA import NVBandwidth as NV
from Benchmarks.NVIDIA import NCCLBandwidth as NCCL
from Benchmarks.NVIDIA import FlashAttention as FA
from Benchmarks.NVIDIA import FIO
from Benchmarks.NVIDIA import CPUStream as CPU
from Benchmarks.NVIDIA import Multichase as Multichase
from Benchmarks.NVIDIA import LLMBenchmark as llmb
from Benchmarks.NVIDIA import LLAMA3Run as llama3pre
from Infra import tools
from prettytable import PrettyTable

host_name = tools.get_hostname()
current = os.getcwd()
tools.create_dir("Outputs")

def get_system_specs():
    results = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name,vbios_version,driver_version,memory.total", "--format=csv"], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output = results.stdout.decode('utf-8').split('\n')[1].split(",")
    if not os.path.exists(current +  "/Outputs/" + host_name + "_summary.md"):
        table = PrettyTable([" ", output[0]])
        table.add_row(["VBIOS", output[1]])
        table.add_row(["driver version", output[2]])
        table.add_row(["GPU memory capacity", output[3]])
        
        results = subprocess.run("nvcc --version | grep release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        cuda_version = results.stdout.decode('utf-8').split(",")[1].strip().split(" ")[1]
        table.add_row(["CUDA version", cuda_version])
    
        if output[0].strip() != "NVIDIA Graphics Device" or "GB200" in output[0]:
            results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            ubuntu = results.stdout.decode('utf-8').strip().split("\t")[1]
            table.add_row(["ubuntu version", ubuntu])
            results = subprocess.run("pip list | grep 'torch '", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            pyt = results.stdout.decode('utf-8').strip().split(" ")[-1]
            table.add_row(["pytorch", pyt])
        print(table)
        tools.export_markdown(output[0].strip() + " Benchmarking Guide", "", table)
    return output[0].strip()

def run_CublasLt():
    test = gemm.GEMMCublastLt("config.json",sku_name)
    test.build()
    test.run_model_sizes()

def run_HBMBandwidth():
    if "GB200" in sku_name:
        print("HBM bandwidth Test not supported on GB200 yet")
        return
    test = HBM.HBMBandwidth("config.json", sku_name)
    test.build()
    test.run()

def run_NVBandwidth():
    test = NV.NVBandwidth("config.json", sku_name)
    test.build()
    test.run()

def run_NCCLBandwidth():
    test = NCCL.NCCLBandwidth("config.json", sku_name)
    test.build()
    test.run()

def run_FlashAttention():
    test = FA.FlashAttention("config.json", sku_name)
    test.run()

def run_Multichase():
    test = Multichase.Multichase("config.json", sku_name)
    test.build()
    test.run()

def run_CPUStream():
    test = CPU.CPUStream("config.json", sku_name)
    test.build()
    test.run()

def run_FIO():
    test = FIO.FIO("config.json", sku_name)
    test.run()

def run_LLMBenchmark():
    test = llmb.LLMBenchmark("config.json", current, sku_name)
    test.install_requirements()
    test.prepare_datasets()
    test.download_models()
    test.run_benchmark()

def run_LLAMA3Pretrain(model_size="8b"):
    test = llama3pre.LLAMA3Pretraining("config.json", sku_name, model_size)
    test.run()

sku_name = get_system_specs()
arguments = []
match = False
for arg in sys.argv:
    arguments.append(arg.lower())

if ("gemm" in arguments):
    match = True
    run_CublasLt()
    os.chdir(current)

if ("nccl" in arguments):
    match = True
    run_NCCLBandwidth()
    os.chdir(current)

if ("hbm" in arguments):
    match = True
    run_HBMBandwidth()
    os.chdir(current)

if ("nv" in arguments):
    match = True
    run_NVBandwidth()
    os.chdir(current)

if ("fa"  in arguments):
    match = True
    run_FlashAttention()
    os.chdir(current)

if ("multichase" in arguments):
    match = True
    run_Multichase()
    os.chdir(current)

if ("cpustream" in arguments):
    match = True
    run_CPUStream()
    os.chdir(current)

if ("fio" in arguments):
    match = True
    run_FIO()
    os.chdir(current)

if ("llm" in arguments):
    match = True
    run_LLMBenchmark()
    os.chdir(current)

if ("llama_8b_pretrain" in arguments):
    match = True
    run_LLAMA3Pretrain("8b")
    os.chdir(current)

if ("llama_3b_pretrain" in arguments):
    match = True
    run_LLAMA3Pretrain("3b")
    os.chdir(current)

if ("all" in arguments):
    match = True
    run_CublasLt()
    os.chdir(current)
    run_NCCLBandwidth()
    os.chdir(current)
    run_Multichase()
    os.chdir(current)
    run_CPUStream()
    os.chdir(current)
    run_HBMBandwidth()
    os.chdir(current)
    run_NVBandwidth()
    os.chdir(current)
    run_FIO()
    os.chdir(current)
    run_FlashAttention()
    os.chdir(current)
    run_LLMBenchmark()
    os.chdir(current)
    run_LLAMA3Pretrain("8b")
    os.chdir(current)
    run_LLAMA3Pretrain("3b")

os.environ["BM_UPLOAD"] = "0"
if not match:
    print("Usage: python3 NVIDIA_runner.py [arg]\n   or: python3 NVIDIA_runner.py [arg1] [arg2] ... to run more than one test e.g python3 NVIDIA_runner.py hbm nccl\nArguments are as follows, and are case insensitive:\nAll tests:  all\nCuBLASLt GEMM:  gemm\nNCCL Bandwidth: nccl\nHBMBandwidth:   hbm\nNV Bandwidth:   nv\nFIO Tests:   fio\nFlash Attention: fa\n   LLM Inference Workloads: llm\nCPU Stream: cpustream\nMultichase:  multichase\nLLAMA 8B Pretrain:  llama_8b_pretrain\nLLAMA 3B Pretrain: llama_3b_pretrain")
