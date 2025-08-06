import subprocess
import os
import docker
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
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'name': 'flash_attention',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'tty': True,
            'detach': True,
            'auto_remove': True
        }

        # Creates new Docker container
        print("Pulling docker container powderluv/vllm_dev_channel:20240927...")
        self.container = client.containers.run('powderluv/vllm_dev_channel:20240927', **docker_run_options)
        print(f"Created Docker Container ID: {self.container.id}")

    def run(self):
        current = os.getcwd()
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run('git clone https://github.com/Dao-AILab/flash-attention.git',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))

        build_path = os.path.join(current, 'flash-attention')
        os.chdir(build_path)

        results = subprocess.run('git checkout 418d677',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))

        self.create_container()
        print("Running Flash Attention...")
        res = self.container.exec_run(f'bash -c "python3 {self.dir_path}/flash-attention/benchmarks/benchmark_flash_attention.py | grep -A 2 "batch_size=2, seqlen=8192 ###""')
        tools.write_log(res.output.decode('utf-8'))
        self.container.kill()

        table = PrettyTable(["causal", "headdim", "Flash2 total (TFLOPs)", "Pytorch total (TFLOPs)"])
        for m in re.findall(r"causal=(\w+), headdim=(\d+).*?fwd \+ bwd: ([\d.]+).*?fwd \+ bwd: ([\d.]+)", res.output.decode('utf-8'), re.DOTALL):
            table.add_row([m[0], int(m[1]), float(m[2]), float(m[3])])
            if "True" in str(m[0]) and int(m[1]) == 128:
                entry = tools.create_bm_entry("batch_size 2 x seq_len 8192", self.name, self.machine_name, str(m[2]))
                print(entry)
        print(table)
        tools.export_markdown("Flash Attention 2", "The performance (in TFLOPS), in table below, represents the performance for a batch size of 2, and a sequence length of 8192.", table)
