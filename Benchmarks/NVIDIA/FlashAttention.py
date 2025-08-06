import subprocess
import re
import os
from prettytable import PrettyTable
from Infra import tools

class FlashAttention:
    def __init__(self, path:str, machine: str):
        self.name='FlashAttention'
        self.machine_name = machine
        self.buffer = []

    def run(self):
        current = os.getcwd()
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run('git clone https://github.com/Dao-AILab/flash-attention.git',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        build_path = os.path.join(current, 'flash-attention/benchmarks')
        os.chdir(build_path)

        print("Running Flash Attention with batch size=2, seqlen=8192...")
        results = subprocess.run('python3 benchmark_flash_attention.py | grep -A 2 "batch_size=2, seqlen=8192 ###"',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        os.chdir(current)

        table = PrettyTable(["causal", "headdim", "Flash2 total (TFLOPs)", "Pytorch total (TFLOPs)"])
        for m in re.findall(r"causal=(\w+), headdim=(\d+).*?fwd \+ bwd: ([\d.]+).*?fwd \+ bwd: ([\d.]+)", results.stdout.decode('utf-8'), re.DOTALL):
            table.add_row([m[0], int(m[1]), float(m[2]), float(m[3])])
            if "True" in str(m[0]) and int(m[1]) == 128:
                entry = tools.create_bm_entry("batch_size 2 x seq_len 8192", self.name, self.machine_name, str(m[2]))
                print(entry)
        print(table)
        tools.export_markdown("Flash Attention 2", "The performance (in TFLOPS), in table below, represents the performance for a batch size of 2, and a sequence length of 8192.", table)
