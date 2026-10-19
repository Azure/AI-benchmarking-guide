import subprocess
import os
import re
from prettytable import PrettyTable
from Infra import tools

class FlashAttention:
    def __init__(self, path:str, machine: str):
        self.name='FlashAttention'
        self.machine_name = machine
        self.dir_path = path
        self.container = None

    def create_container(self):
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': tools.container_volumes(self.dir_path),
            'tty': True,
            'detach': True,
            'auto_remove': True
        }

        # Creates new Docker container
        tools.progress("Pulling docker container powderluv/vllm_dev_channel:20240927...")
        self.container = tools.start_container('powderluv/vllm_dev_channel:20240927', **docker_run_options)
        tools.progress(f"Created Docker Container ID: {self.container.id}")

    def run(self):
        current = os.getcwd()
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_command('git clone https://github.com/Dao-AILab/flash-attention.git',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        build_path = os.path.join(current, 'flash-attention')
        os.chdir(build_path)

        results = tools.run_command('git checkout 418d677',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            with tools.managed_container(self):
                self.create_container()
                tools.progress("Running Flash Attention...")
                output = tools.run_container_command(
                    self.container,
                    ["python3", f"{self.dir_path}/flash-attention/benchmarks/benchmark_flash_attention.py"]
                )
        finally:
            os.chdir(current)

        lines = output.splitlines()
        selected = []
        for i, line in enumerate(lines):
            if "batch_size=2, seqlen=8192 ###" in line:
                selected.extend(lines[i:i + 3])

        table = PrettyTable(["causal", "headdim", "Flash2 total (TFLOPs)", "Pytorch total (TFLOPs)"])
        matches = re.findall(r"causal=(\w+), headdim=(\d+).*?fwd \+ bwd: ([\d.]+).*?fwd \+ bwd: ([\d.]+)", "\n".join(selected), re.DOTALL)
        if not matches:
            raise tools.runtime.OutputError("No FlashAttention measurements for batch_size=2, seqlen=8192")
        for m in matches:
            table.add_row([m[0], int(m[1]), float(m[2]), float(m[3])])
        tools.detail(table)
        tools.export_markdown("Flash Attention 2", "The performance (in TFLOPS), in table below, represents the performance for a batch size of 2, and a sequence length of 8192.", table)
