import os
from Infra import tools
from prettytable import PrettyTable
import docker

class GEMMHipBLAS:
    def __init__(self, path: str, dir_path: str, machine: str, i: int = 1000, w: int = 10000):
        self.name = "GEMMHipBLAS"
        self.datatype = "FP8"
        self.dir_path = dir_path
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
        self.container = None

    def create_container(self):
        client = docker.from_env()
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
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'tty': True,
            'detach': True
        }
        # Creates new Docker container
        print("Pulling docker container rocm/vllm-dev:main...")
        self.container = client.containers.run('rocm/vllm-dev:main', **docker_run_options)
        print(f"Launched Docker Container ID: {self.container.id}")

    def build(self):
        path = "hipBLASLt"
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/ROCm/hipBLASLt " + self.dir_path + "/hipBLASLt"
            results = self.container.exec_run(clone_cmd, stderr=True)
            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/hipBLASLt && git checkout a11ccf64efcd818106dbe37768f69dfcc0a7ff22"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                return

            results = self.container.exec_run(f'sudo apt-get -y update', stderr=True)
            tools.write_log(results.output.decode('utf-8'))
            results = self.container.exec_run(f'sudo apt -y install llvm-dev', stderr=True)
            tools.write_log(results.output.decode('utf-8'))
            print("Building hipBLAS Library...")
            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/hipBLASLt && ./install.sh -dc -a gfx942"', stderr=True)
            tools.write_log(results.output.decode('utf-8'))

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        print("Running HipBLAS...")
        m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]

        for i in range(len(m_dims)):
            hipblas_cmd = 'cd ' + self.dir_path + '/Benchmarks/AMD && ./hipBLAS_runner.sh ' + str(m_dims[i]) + ' ' +  str(n_dims[i]) + ' ' + str(k_dims[i])
            results = self.container.exec_run(f'/bin/sh -c ' + '"' + hipblas_cmd + '"')
            tools.write_log(results.output.decode('utf-8'))

        with open(self.dir_path + '/Outputs/GEMMHipBLAS_results.txt', 'r') as resFile:
            table1 = PrettyTable()
            table1.field_names = ["M","N","K","TFLOPS"]
            for line in resFile:
                l = line.strip()
                if l[0] == "T":
                    l = l.split(',')
                    m = l[4]
                    n = l[5]
                    k = l[6]
                    tflops = float(l[-3])/1000
                    table1.add_row([m,n,k,tflops])
                    print(tools.create_bm_entry(m+"x"+n+"x"+k, self.name, self.machine_name, tflops))

        print(table1)
        tools.export_markdown("GEMM HipBLASLt", "The results shown below are with random initialization (best representation of real-life workloads) " + self.datatype +  ", and " + str(self.w) + " warmup iterations.", table1)
        results = self.container.exec_run(f'/bin/sh -c "rm {self.dir_path}/Outputs/GEMMHipBLAS_results.txt"', stderr=True)
        self.container.kill()
