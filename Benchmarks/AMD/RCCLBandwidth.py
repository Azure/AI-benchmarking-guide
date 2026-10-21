from prettytable import PrettyTable
from Infra import tools

_TESTS_REVISION = "ae3e6357cb3514d5649b3b5dfaa80cb2d2af3532"
_VERSION_QUERY = """
import ctypes

library = ctypes.CDLL("/usr/local/lib/librccl.so")
library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
library.ncclGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
status = library.ncclGetVersion(ctypes.byref(version))
if status or version.value <= 0:
    raise SystemExit(f"Invalid RCCL version: status={status}, version={version.value}")
print(f"{version.value // 10000}.{version.value % 10000 // 100}.{version.value % 100}")
"""

class RCCLBandwidth:
    def __init__(self, config_path:str, dir_path:str, machine: str):
        self.name='RCCLBandwidth'
        self.machine_name = machine
        self.dir_path = dir_path
        self.container = None
        self.tests_path = tools.get_output_dir() / "rccl-tests"
        self.test_executable = self.tests_path / "build" / "all_reduce_perf"

    def create_container(self):
        docker_run_options = tools.amd_container_options(self.dir_path, entrypoint="/bin/bash")

        tools.progress(f"Preparing Docker image {tools.AMD_BENCHMARK_IMAGE}...")
        self.container = tools.start_container(tools.AMD_BENCHMARK_IMAGE, **docker_run_options)
        tools.progress(f"Docker Container ID: {self.container.id}")

    def build(self):
        for dependency in ("/usr/local/include/rccl/rccl.h", "/usr/local/lib/librccl.so"):
            tools.run_container_command(self.container, ["test", "-r", dependency])
        version = tools.run_container_command(self.container, ["python3", "-c", _VERSION_QUERY])
        tools.progress(f"Using container-installed RCCL {version.strip()}; skipping RCCL source build")
        tools.progress(f"Building RCCL AllReduce test at revision {_TESTS_REVISION}...")
        tools.run_container_command(self.container, [
            "git", "clone", "https://github.com/ROCm/rccl-tests.git", str(self.tests_path),
        ])
        tools.run_container_command(self.container, [
            "git", "-C", str(self.tests_path), "checkout", "--detach", _TESTS_REVISION,
        ])
        tools.run_container_command(self.container, [
            "make", "-C", str(self.tests_path / "src"), str(self.test_executable),
            f"BUILDDIR={self.tests_path / 'build'}", "ROCM_PATH=/opt/rocm",
            "HIP_HOME=/opt/rocm", "NCCL_HOME=/usr/local", "CUSTOM_RCCL_LIB=/usr/local/lib",
        ])
        tools.run_container_command(self.container, ["test", "-x", str(self.test_executable)])

    def run(self):
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        runs = ["Tree", "Ring", "NVLS", "NVLSTree"]
        tools.progress("Running RCCL AllReduce...")
        for run in runs:
            tools.progress(f"RCCL: measuring {run}")
            output = tools.run_container_command(self.container, [
                "env", f"NCCL_ALGO={run}", str(self.test_executable),
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
