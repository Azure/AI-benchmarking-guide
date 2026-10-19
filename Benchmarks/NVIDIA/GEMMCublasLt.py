import json
import os
import shlex
import subprocess
import datetime
import time
from Infra import tools
from prettytable import PrettyTable

class GEMMCublastLt:
    def __init__(self, path: str, machine: str, b: int = 1, i: int = 1000, w: int = 10000):
        self.name = "GEMMCublasLt"
        config = self.get_config(path)
        self.datatype = self.config_conversion(config)
        self.b = b
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
       
        # A100 does not support fp8
        if "A100" in machine:
            self.datatype = "fp16"

    def get_config(self, path: str):
        return tools.load_config(path, self.name)

    def config_conversion(self, config):
        datatype = config.get("datatype")
        if datatype not in ("fp8e4m3", "fp4e2m1", "fp16"):
            raise tools.runtime.ConfigurationError(f"Unsupported GEMM datatype: {datatype}")
        return datatype

    def build(self):
        bindir = tools.create_dir("bin")
        self.bindir = bindir
        path = "superbenchmark"
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_command(['git', 'clone', 'https://github.com/gitaumark/superbenchmark', path])
            
        if self.datatype == "fp4e2m1":
            results = tools.run_command('cd superbenchmark && git checkout fp4', shell=True)
        else:
            results = tools.run_command('cd superbenchmark && git checkout main', shell=True)
        
        current = os.getcwd()
        build_path = os.path.join(
            current,
            "superbenchmark/superbench/benchmarks/micro_benchmarks/cublaslt_gemm",
        )
        os.chdir(build_path)

        results = tools.run_command(['cmake', '-S', './'])

        results = tools.run_command(['make'])
        tools.detail(results.stderr.decode('utf-8'))
        results = tools.run_command(['mv', 'cublaslt_gemm', bindir])
        os.chdir(current)

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        tools.progress("Running CublasLt with datatype " + self.datatype + "...")
        current = os.getcwd()
        if self.datatype == "fp8e4m3":
            m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]
        elif self.datatype == "fp4e2m1":
            m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 768]
        else:
            m_dims = [1024, 2048, 4096, 8192, 16384, 1024, 6144, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 2145, 12288, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 1024, 12288, 768]
        os.chdir(self.bindir)
        buffer = []
        for i in range(len(m_dims)):
            tools.progress(f"GEMM: measurement {i + 1}/{len(m_dims)}")
            results = tools.run_command(
                [
                    "./cublaslt_gemm",
                    "-m",
                    str(m_dims[i]),
                    "-n",
                    str(n_dims[i]),
                    "-k",
                    str(k_dims[i]),
                    "-b",
                    str(self.b),
                    "-i",
                    str(self.i),
                    "-w",
                    str(self.w),
                    "-t",
                    self.datatype,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log = results.stdout.decode('utf-8').split()
            if len(log) != 6:
                raise tools.runtime.OutputError(f"Expected six GEMM result fields, received {len(log)}")
            buffer.append(log)
        table1 = PrettyTable()
        table1.field_names = ["M", "N", "K", "Batch Size", "Time(us)", "TFLOPS"]
        for item in buffer:
            table1.add_row(item)
        tools.detail(table1)
        tools.export_markdown("GEMM CuBLASLt", "The results shown below are with random initialization (best representation of real-life workloads) " + self.datatype +  ", and " + str(self.w) + " warmup iterations.", table1)
        os.chdir(current)
