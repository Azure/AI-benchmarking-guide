import json
import os
import csv
import subprocess
import csv
from Infra import tools
from prettytable import PrettyTable

class NCCLBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NCCLBandwidth'
        self.machine_name = machine
        self.buffer = []
        self.algo = "NVLS"

    def build(self):
        current = os.getcwd()
        path ='nccl'
        isdir = os.path.isdir(path)
        if not isdir:
            print("Building NCCL Library...")
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nccl.git', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl')
            os.chdir(build_path)
            results = subprocess.run('make -j src.build', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            os.chdir(current)

        results = subprocess.run('export NCCL_HOME=' + current + '/nccl/build', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results = subprocess.run('export LD_LIBRARY_PATH=' + current + '/nccl/build/lib:$LD_LIBRARY_PATH', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        path ='nccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            print("Building NCCL Test..")
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nccl-tests.git', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl-tests')
            os.chdir(build_path)
            results = subprocess.run(['make'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
        else:
            build_path = os.path.join(current, 'nccl-tests')
            os.chdir(build_path)

    def run(self):
        current = os.getcwd()
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        num_gpus = str(subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')).strip()
        if num_gpus == '4':
            self.algo = "Ring"
        print("Running NCCL AllReduce on " + num_gpus + " GPUs")

        results = subprocess.run('NCCL_ALGO='+ self.algo +' ./build/all_reduce_perf -b 8 -e 8G -f 2 -g ' + num_gpus + ' -n 40 | grep float', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        res = results.stdout.decode('utf-8').split('\n')
        log = []
        for line in res:
            line = line.split()
            if len(line) == 13:
                log.append(line[11])

        buffer.append(log)
        table1 = PrettyTable()
        runs = ["Message Size", "Bandwidth (" + self.algo + ")"]
        for i in range(len(buffer)):
            table1.add_column(runs[i], buffer[i])
        print(table1)
        tools.export_markdown("NCCL Bandwidth", "The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce (Ring algorithm) tests in-place operations, varying from 1KB to 8GB of data.", table1)
        os.chdir(current)
