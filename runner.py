import json
import os
import subprocess
import GEMMCublasLt as gemm
import HBMBandwidth as HBM
import NVBandwidth as NV
import NCCLBandwidth as NCCL
import FlashAttention as FA
from Infra import tools
import LLMBenchmark as llmb

machine_name = ""
current = os.getcwd()
tools.create_dir("Outputs")


def get_system_specs():

    file = open("Outputs/system_specs.txt", "w")

    results = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name,vbios_version,driver_version,memory.total", "--format=csv"], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output = results.stdout.decode('utf-8').split('\n')[1].split(",")
    file.write("GPU name     : "+ output[0]+"\n")
    file.write("VBIOS    : "+ output[1]+"\n")
    file.write("driver version   : "+ output[2]+"\n")
    file.write("GPU memory capacity  : "+ output[3]+"\n")
    

    results = subprocess.run("nvcc --version | grep release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    cuda_version = results.stdout.decode('utf-8').split(",")[1].strip().split(" ")[1]
    file.write("CUDA version     : "+cuda_version+"\n")

    results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    ubuntu = results.stdout.decode('utf-8').strip().split("\t")[1]
    file.write("ubuntu version   : "+ubuntu+"\n")

    results = subprocess.run("pip3 show torch | grep Version", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    file.write("pytorch version  : "+ results.stdout.decode('utf-8').split(" ")[1].strip()+"\n")

    results = subprocess.run("grep 'stepping\|model\|microcode' /proc/cpuinfo | grep microcode", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    microcode = results.stdout.decode('utf-8').split("\n")[0]
    file.write(microcode+"\n")

    results = subprocess.run("grep 'stepping\|model\|microcode' /proc/cpuinfo | grep name", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    file.write(results.stdout.decode('utf-8').split("\n")[0]+"\n")

    results = subprocess.run("grep 'cores\|model\|microcode' /proc/cpuinfo | grep cores", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    file.write(results.stdout.decode('utf-8').split("\n")[0])
    file.close()
    return output[0].strip()


def run_CublasLt():
    test = gemm.GEMMCublastLt("config.json",machine_name)

    # builds the CublasLt binary 
    test.build()

    # generates the table with predetermined m,n,k values
    test.run_model_sizes()

    # generates power, clock, and gpu temperature plots
    test.run_nvml()

    # runs GEMM sweep and generates shmoo plots (takes about 20 minutes)
    # test.run_shmoo()
    

def run_HBMBandwidth():
    test = HBM.HBMBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_NVBandwidth():
    test = NV.NVBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_NCCLBandwidth():
    test = NCCL.NCCLBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_FlashAttention():
    test = FA.FlashAttention("config.json", machine_name)
    test.run()

def run_LLMBenchmark():
    test = llmb.LLMBenchmark("config.json", current, machine_name)
    test_cublaslt = gemm.GEMMCublastLt("config.json",machine_name)
    test_cublaslt.build()
    test.create_container()
    test.download_models()
    test.build_engines()
    test.run_benchmark()

machine_name = get_system_specs()

# Tests (comment out the ones you don't want to run)
run_CublasLt()
os.chdir(current)
run_NCCLBandwidth()
os.chdir(current)
run_HBMBandwidth()
os.chdir(current)
run_NVBandwidth()
os.chdir(current)
run_FlashAttention()
run_LLMBenchmark()
