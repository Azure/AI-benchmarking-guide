import os
import shlex
from prettytable import PrettyTable
from Infra import tools

class RCCLBandwidth:
    def __init__(self, config_path:str, dir_path:str, machine: str):
        self.name='RCCLBandwidth'
        self.machine_name = machine
        self.dir_path = dir_path
        self.container = None

    def create_container(self):
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'entrypoint': '/bin/bash',
            'network': 'host',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': tools.container_volumes(self.dir_path),
            'tty': True,
            'detach': True
        }

        # Creates new Docker container from https://hub.docker.com/r/rocm/pytorch/tags
        tools.progress("Pulling docker container rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue...")
        self.container = tools.start_container('rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue', **docker_run_options)
        tools.progress(f"Docker Container ID: {self.container.id}")

    def build(self):
        path ='rccl'
        isdir = os.path.isdir(path)
        if not isdir:
            tools.progress("Building RCCL Library...")
            tools.run_container_command(self.container, ["git", "clone", "https://github.com/ROCm/rccl.git", self.dir_path + "/rccl"])
            tools.run_container_command(self.container, ["bash", "-o", "pipefail", "-c",
                f"cd {shlex.quote(self.dir_path + '/rccl')} && cmake . && make"])

        path ='rccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            tools.progress("Building RCCL Tests...")
            tools.run_container_command(self.container, ["git", "clone", "https://github.com/ROCm/rccl-tests.git", self.dir_path + "/rccl-tests"])
            directory = shlex.quote(self.dir_path + "/rccl-tests")
            library = shlex.quote(self.dir_path + "/rccl")
            tools.run_container_command(self.container, ["bash", "-o", "pipefail", "-c",
                f"cd {directory} && make HIP_HOME=/opt/rocm NCCL_HOME={library} CUSTOM_RCCL_LIB={library}/librccl.so && make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm NCCL_HOME={library}"])

    def run(self):
        with tools.managed_container(self):
            self._run()

    def _run(self):
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        runs = ["Tree", "Ring", "NVLS", "NVLSTree"]
        tools.progress("Running RCCL AllReduce...")
        for run in runs:
            tools.progress(f"RCCL: measuring {run}")
            output = tools.run_container_command(self.container, [
                "env", f"NCCL_ALGO={run}", self.dir_path + "/rccl-tests/build/all_reduce_perf",
                "-b", "8", "-e", "8G", "-f", "2", "-g", "8", "-n", "40"])
            res = output.splitlines()
            log = []
            for line in res:
                line = line.split()
                if len(line) == 13 and line[2] == "float":
                    log.append(line[11])
            if len(log) != len(buffer[0]):
                raise tools.runtime.OutputError(f"Expected {len(buffer[0])} RCCL measurements for {run}, received {len(log)}")
            buffer.append(log)
        table1 = PrettyTable()
        runs = ["Message Size", "Tree", "Ring", "NVLS", "NVLSTree"]

        for i in range(len(buffer)):
            table1.add_column(runs[i], buffer[i])
        tools.detail(table1)
        tools.export_markdown("RCCL Bandwidth", "The values (in GB/s) are the bus bandwidth values obtained from the RCCL AllReduce tests with Tree, Ring, NVLS and NVLSTree algos (in-place operations), varying from 1KB to 8GB of data.", table1)
