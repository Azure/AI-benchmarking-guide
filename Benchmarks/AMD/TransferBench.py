import os
import math
import re
import subprocess
import csv
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
        executable = os.path.join(self.dir_path, "TransferBench/build/TransferBench")
        run_cmd = [
            "sudo", executable,
            os.path.join(self.dir_path, "Benchmarks/AMD/transferbench.cfg"),
        ]
        results = tools.run_command(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        measurements = []
        for line in results.stdout.decode("utf-8").splitlines():
            if "=" in line or "sum" not in line:
                continue
            fields = re.split(r"[|\u2502]", line)
            if len(fields) < 2:
                raise tools.runtime.OutputError(f"Invalid TransferBench measurement: {line}")
            executor = re.fullmatch(r"\s*Executor:\s+GPU\s+(\d+)\s*", fields[0])
            if not executor:
                raise tools.runtime.OutputError(f"Invalid TransferBench executor: {line}")
            bandwidth = fields[1].strip()
            match = re.fullmatch(r"(\S+)\s+GB/s", bandwidth)
            if not match:
                raise tools.runtime.OutputError(f"Invalid TransferBench bandwidth: {line}")
            try:
                value = float(match.group(1))
            except ValueError as exc:
                raise tools.runtime.OutputError(f"Invalid TransferBench bandwidth: {line}") from exc
            if not math.isfinite(value) or value <= 0:
                raise tools.runtime.OutputError(f"Invalid TransferBench bandwidth: {line}")
            measurements.append((int(executor.group(1)), bandwidth))
        tests = (
            ("Host to Device memcpy", 0),
            ("Device to Host memcpy", 0),
        )
        if len(measurements) != len(tests):
            raise tools.runtime.OutputError(
                f"Expected {len(tests)} TransferBench measurements, received {len(measurements)}"
            )
        table = PrettyTable(["Test", "Result"])
        for (label, expected_executor), (executor, bandwidth) in zip(tests, measurements):
            if executor != expected_executor:
                raise tools.runtime.OutputError(
                    f"Expected GPU {expected_executor} executor for {label}, received GPU {executor}"
                )
            table.add_row([label, bandwidth])
        matrix = self._run_gpu_matrix(executable)
        tools.detail(table)
        tools.detail(matrix)
        tools.export_markdown("TransferBench", "Host transfer bandwidth in GB/s.", table)
        tools.export_markdown(
            "Device to Device Bidirectional Read Total",
            "GPU-kernel bandwidth in GB/s using 32 CUs per destination GPU. Each cell "
            "sums the two simultaneous read directions for that GPU pair. Pairs are "
            "tested separately, not all-to-all concurrently. '-' means self-transfer "
            "or an unavailable peer path; these are not copy-engine measurements.",
            matrix,
        )

    def _run_gpu_matrix(self, executable):
        tools.progress("Measuring the full GPU-to-GPU bidirectional read matrix...")
        result = tools.run_command([
            "sudo", "env", "-u", "NUM_GPU_DEVICES", "P2P_MODE=2", "NUM_CPU_DEVICES=0",
            "NUM_GPU_SE=32", "USE_REMOTE_READ=1", "USE_GPU_DMA=0", "USE_GPU_TDM=0",
            "SHOW_ITERATIONS=0", "OUTPUT_TO_CSV=1", executable, "p2p",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_ids = None
        totals = {}
        for row in csv.reader(result.stdout.decode("utf-8").splitlines()):
            fields = [field.strip() for field in row]
            if fields and not fields[-1]:
                fields.pop()
            if not fields:
                continue
            if fields[0] == "SRC\\DST":
                if gpu_ids is not None:
                    raise tools.runtime.OutputError("Duplicate TransferBench matrix header")
                columns = [re.fullmatch(r"GPU\s+(\d+)", field) for field in fields[1:]]
                if len(columns) < 2 or any(column is None for column in columns):
                    raise tools.runtime.OutputError("Expected a TransferBench matrix with at least two GPUs")
                gpu_ids = [int(column.group(1)) for column in columns]
                if len(set(gpu_ids)) != len(gpu_ids):
                    raise tools.runtime.OutputError("Duplicate TransferBench matrix GPU columns")
                continue
            label = re.fullmatch(r"GPU\s+(\d+)\s+<->", fields[0])
            if label is None:
                continue
            gpu = int(label.group(1))
            if gpu_ids is None or gpu not in gpu_ids or gpu in totals or len(fields) != len(gpu_ids) + 1:
                raise tools.runtime.OutputError(f"Invalid TransferBench matrix row: {row}")
            values = []
            for peer, bandwidth in zip(gpu_ids, fields[1:]):
                if bandwidth == "N/A":
                    if peer != gpu:
                        tools.logger.warning("TransferBench peer path unavailable: GPU %s <-> GPU %s", gpu, peer)
                    values.append("-")
                    continue
                try:
                    value = float(bandwidth)
                except ValueError as exc:
                    raise tools.runtime.OutputError(f"Invalid TransferBench matrix bandwidth: {row}") from exc
                if not math.isfinite(value) or value <= 0 or gpu == peer:
                    raise tools.runtime.OutputError(f"Invalid TransferBench matrix bandwidth: {row}")
                values.append(bandwidth)
            totals[gpu] = values
        if gpu_ids is None or set(totals) != set(gpu_ids):
            raise tools.runtime.OutputError("TransferBench returned an incomplete GPU bidirectional matrix")
        matrix = PrettyTable(["GPU", *[f"GPU {gpu}" for gpu in gpu_ids]])
        for gpu in gpu_ids:
            matrix.add_row([f"GPU {gpu}", *totals[gpu]])
        return matrix
