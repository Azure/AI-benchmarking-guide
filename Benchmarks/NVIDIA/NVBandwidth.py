import json
import subprocess
import os
from Infra import tools
from prettytable import PrettyTable

class NVBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NVBandwidth'
        self.machine_name = machine

    def build(self):
        current = os.getcwd()
        path ='nvbandwidth'
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_command(['git', 'clone', 'https://github.com/NVIDIA/nvbandwidth', path])
        build_path = os.path.join(current, 'nvbandwidth')
        os.chdir(build_path)
        results = tools.run_command(['sed', '-i', '2i\\set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)', 'CMakeLists.txt'])

        if os.path.exists("/.dockerenv"):
            results = tools.run_command('apt update && ./debian_install.sh', shell=True)
        else:
            results = tools.run_command('sudo apt update && sudo ./debian_install.sh', shell=True)
        os.chdir(current)

    def run(self):
        current = os.getcwd()
        os.chdir(os.path.join(current, 'nvbandwidth'))
        tools.progress("Running NVBandwidth...")
        results = tools.run_command('./nvbandwidth -t device_to_host_memcpy_ce host_to_device_memcpy_ce device_to_device_bidirectional_memcpy_read_ce', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = results.stdout.decode('utf-8')
        os.chdir(current)

        item = "\n".join(log[log.find("device_to_host_memcpy_ce"):].strip().splitlines()[:-2])
        self.format_output(item)
        os.chdir(current)
        
    def format_output(self, text):
        tables, current = [], []
        for line in text.splitlines():
            if not line.strip():
                if current:
                    tables.append(current)
                    current = []
                continue
            if 'memcpy' in line:
                continue
            row = [x.strip() for x in line.split() if x.strip()]
            current.append([int(float(x)) if x.replace('.', '', 1).isdigit() else "-" for x in row])
        if current:
            tables.append(current)
        if len(tables) < 5 or any(len(tables[i]) < 2 for i in (0, 1, 4)):
            raise tools.runtime.OutputError("NVBandwidth returned incomplete transfer matrices")
        result = [tables[0], tables[1], tables[4]]
        labels = ["Device to Host memcpy", "Host to Device memcpy", "Device to Device Bidirectional memcpy Total"]
        for i in range(len(result)):
            result[i][0].insert(0, " ")
            t = PrettyTable(result[i][0])
            for j in range(1, len(result[i])):
                t.add_row(result[i][j])
            tools.progress(labels[i])
            tools.detail(t)

            if i == 0:
                tools.export_markdown("NV Bandwidth", labels[i], t)
            else:
                tools.export_markdown(None, labels[i], t)
                
            
