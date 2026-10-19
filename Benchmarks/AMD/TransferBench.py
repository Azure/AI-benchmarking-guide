import os
import subprocess
from prettytable import PrettyTable
from Infra import tools

class TransferBench:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "TransferBench"
        self.machine_name = machine
        self.dir_path = dir_path

    def build(self):
        path = "TransferBench"
        isdir = os.path.isdir(path)
        if not isdir:
            tools.progress("Building TransferBench...")
            clone_cmd = "git clone https://github.com/ROCm/TransferBench.git " + self.dir_path + "/TransferBench"
            results = tools.run_command(clone_cmd, shell=True)
            results = tools.run_command('mkdir ' + self.dir_path + '/TransferBench/build', shell=True)

            results = tools.run_command('cd ' + self.dir_path + '/TransferBench/build && CXX=/opt/rocm/bin/hipcc cmake .. && make', shell=True)

    def run(self):
        tools.progress("Running TransferBench...")
        run_cmd = "sudo " + self.dir_path + "/TransferBench/build/TransferBench " + self.dir_path + "/Benchmarks/AMD/transferbench.cfg | grep -v '=' | grep 'sum'"
        results = tools.run_command(run_cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        log = results.stdout.decode("utf-8").split("|")
        if len(log) < 6 or not log[1].strip() or not log[5].strip():
            raise tools.runtime.OutputError("TransferBench returned incomplete transfer measurements")
        table = PrettyTable(["Test", "Result"])
        table.add_row(["Host to Device memcpy", log[1].strip()])
        table.add_row(["Device to Host memcpy", log[5].strip()])
        tools.detail(table)
        tools.export_markdown("TransferBench", "TransferBench Results in GB/s", table)
