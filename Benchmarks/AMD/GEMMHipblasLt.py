import os
import shlex
from Infra import tools
from prettytable import PrettyTable

class GEMMHipBLAS:
    def __init__(self, path: str, dir_path: str, machine: str, i: int = 1000, w: int = 10000):
        self.name = "GEMMHipBLAS"
        self.datatype = "FP8"
        self.dir_path = dir_path
        self.i = i
        self.w = w
        self.machine_name = machine
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
        # Creates new Docker container
        tools.progress("Pulling docker container rocm/vllm-dev:main...")
        self.container = tools.start_container('rocm/vllm-dev:main', **docker_run_options)
        tools.progress(f"Launched Docker Container ID: {self.container.id}")

    def build(self):
        path = "hipBLASLt"
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/ROCm/hipBLASLt " + self.dir_path + "/hipBLASLt"
            tools.run_container_command(self.container, ["git", "clone", "https://github.com/ROCm/hipBLASLt", self.dir_path + "/hipBLASLt"])
            tools.run_container_command(self.container, ["git", "-C", self.dir_path + "/hipBLASLt", "checkout", "a11ccf64efcd818106dbe37768f69dfcc0a7ff22"])
            tools.run_container_command(self.container, ["sudo", "apt-get", "-y", "update"])
            tools.run_container_command(self.container, ["sudo", "apt", "-y", "install", "llvm-dev"])
            tools.progress("Building hipBLAS Library...")
            tools.run_container_command(self.container, ["/bin/bash", "-o", "pipefail", "-c",
                f"cd {shlex.quote(self.dir_path + '/hipBLASLt')} && ./install.sh -dc -a gfx942"])

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        with tools.managed_container(self):
            self._run_model_sizes()

    def _run_model_sizes(self):
        tools.progress("Running HipBLAS...")
        m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]

        for i in range(len(m_dims)):
            tools.progress(f"GEMM: measurement {i + 1}/{len(m_dims)}")
            hipblas_cmd = 'cd ' + shlex.quote(self.dir_path + '/Benchmarks/AMD') + ' && ./hipBLAS_runner.sh ' + str(m_dims[i]) + ' ' + str(n_dims[i]) + ' ' + str(k_dims[i]) + ' ' + str(self.i) + ' ' + str(self.w)
            tools.run_container_command(self.container, ["/bin/sh", "-c", hipblas_cmd])

        results_path = tools.get_output_dir() / "GEMMHipBLAS_results.txt"
        with open(results_path, 'r') as resFile:
            table1 = PrettyTable()
            table1.field_names = ["M","N","K","TFLOPS"]
            for line in resFile:
                l = line.strip()
                if l.startswith("T"):
                    l = l.split(',')
                    m = l[4]
                    n = l[5]
                    k = l[6]
                    tflops = float(l[-3])/1000
                    table1.add_row([m,n,k,tflops])

        if len(table1.rows) != len(m_dims):
            raise tools.runtime.OutputError(f"Expected {len(m_dims)} GEMM measurements, received {len(table1.rows)}")
        tools.detail(table1)
        tools.export_markdown("GEMM HipBLASLt", "The results shown below are with random initialization (best representation of real-life workloads) " + self.datatype +  ", and " + str(self.w) + " warmup iterations.", table1)
