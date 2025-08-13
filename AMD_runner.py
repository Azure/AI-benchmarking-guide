import os
import sys
import subprocess
from Benchmarks.AMD import RCCLBandwidth as RCCL
from Benchmarks.AMD import FlashAttention as FA
from Benchmarks.AMD import HBMBandwidth as HBM
from Benchmarks.AMD import TransferBench as TB
from Benchmarks.AMD import GEMMHipblasLt as GEMM
from Benchmarks.AMD import FIO
from Infra import tools
from Benchmarks.AMD import LLMBenchmark as llmb

current = os.getcwd()
tools.create_dir("Outputs")

def get_system_specs():
    file = open("Outputs/system_specs.txt", "w")

    results = subprocess.run("rocminfo | grep 'ROCk module version'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    rocm_version = results.stdout.decode('utf-8').strip().split(" ")[3]
    file.write("ROCm version     : "+rocm_version+"\n")

    results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    ubuntu = results.stdout.decode('utf-8').strip().split("\t")[1]
    file.write("ubuntu version   : "+ubuntu+"\n")

    results = subprocess.run("grep 'stepping\|model\|microcode' /proc/cpuinfo | grep microcode", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    microcode = results.stdout.decode('utf-8').split("\n")[0]
    file.write(microcode+"\n")

    results = subprocess.run("grep 'stepping\|model\|microcode' /proc/cpuinfo | grep name", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    file.write(results.stdout.decode('utf-8').split("\n")[0]+"\n")

    results = subprocess.run("grep 'cores\|model\|microcode' /proc/cpuinfo | grep cores", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    file.write(results.stdout.decode('utf-8').split("\n")[0])
    file.close()
    return "ND_MI300X_v5"

def run_TransferBench():
    test = TB.TransferBench("config.json", current, machine_name)
    test.build()
    test.run()

def run_GEMMHipBLAS():
    test = GEMM.GEMMHipBLAS("config.json", current, machine_name)
    test.create_container()
    test.build()
    test.run_model_sizes()

def run_RCCLBandwidth():
    test = RCCL.RCCLBandwidth("config.json", current, machine_name)
    test.create_container()
    test.build()
    test.run()

def run_FlashAttention():
    test = FA.FlashAttention(current, machine_name)
    test.run()
    os.chdir(current)

def run_FIO():
    test = FIO.FIO(current, machine_name)
    test.run()

def run_HBMBandwidth():
    test = HBM.HBMBandwidth("config.json", current, machine_name)
    test.build()
    test.run()

def run_LLMBenchmark():
    test = llmb.LLMBenchmark("config.json", current, machine_name)
    test.create_container()
    test.run_benchmark()

machine_name = get_system_specs()
arguments = []
match = False
for arg in sys.argv:
    arguments.append(arg.lower())

if ("gemm" in arguments):
    match = True
    run_GEMMHipBLAS()

if ("rccl" in arguments):
    match = True
    run_RCCLBandwidth()

if ("hbm" in arguments):
    match = True
    run_HBMBandwidth()

if ("transfer" in arguments):
    match = True
    run_TransferBench()

if ("fa" in arguments):
    match = True
    run_FlashAttention()

if ("fio" in arguments):
    match = True
    run_FIO()

if ("llm" in arguments):
    match = True
    run_LLMBenchmark()

if ("all" in arguments):
    match = True
    run_HBMBandwidth()
    run_TransferBench()
    run_RCCLBandwidth()
    run_FIO()
    run_FlashAttention()
    run_LLMBenchmark()
    run_GEMMHipBLAS()

os.environ["BM_UPLOAD"] = "0"
if not match:
    print("Usage: python3 AMD_runner.py [arg]\n   or: python3 AMD_runner.py [arg1] [arg2] ... to run more than one test e.g python3 AMD_runner.py hbm nccl\nArguments are as follows, and are case insensitive:\nAll tests:  all\nROCBLAS GEMM:  gemm\nRCCL Bandwidth: rccl\nHBMBandwidth:   hbm\nTransferbench:   transfer\nFlash Attention: fa\nFIO Tests:   fio\nLLM Inference Workloads: llm")
    
