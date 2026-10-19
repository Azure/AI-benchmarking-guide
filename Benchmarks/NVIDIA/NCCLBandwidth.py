import os
from pathlib import Path
import shlex
import subprocess
import sys
from Infra import tools
from Infra import runtime
from prettytable import PrettyTable


_VERSION_QUERY = """
import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
library.ncclGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
status = library.ncclGetVersion(ctypes.byref(version))
if status:
    raise SystemExit(f"ncclGetVersion failed with code {status}")
print(version.value)
"""


def find_nccl(env, local_build):
    library_dirs = [Path(path).expanduser() for path in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path]
    prefixes = [Path(env[name]).expanduser() for name in ("NCCL_HOME", "NCCL_ROOT") if env.get(name)]
    prefixes.extend(directory.parent for directory in library_dirs)
    prefixes.extend(directory.parent.parent for directory in library_dirs)
    prefixes.extend(Path(path) for path in ("/usr/local/nccl", "/opt/nccl", "/usr/local/cuda", "/usr/local", "/usr"))
    prefixes.extend(Path(path) / "nvidia" / "nccl" for path in sys.path if path)
    prefixes.append(local_build)
    for prefix in dict.fromkeys(prefixes):
        include = prefix / "include"
        if not (include / "nccl.h").is_file():
            continue
        candidates = [prefix / "lib", prefix / "lib64", *sorted(prefix.glob("lib/*-linux-gnu"))]
        candidates.extend(directory for directory in library_dirs
                          if prefix in (directory.parent, directory.parent.parent))
        for library in candidates:
            if (library / "libnccl.so").is_file():
                return include.resolve(), library.resolve()
    return None


class NCCLBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NCCLBandwidth'
        self.machine_name = machine
        self.buffer = []
        self.algo = "NVLS"
        self.env = os.environ.copy()
        self.root = Path.cwd()
        self.using_installed = False
        self.test_executable = self.root / "nccl-tests" / "build" / "all_reduce_perf"

    def build(self):
        installation = find_nccl(self.env, self.root / "nccl" / "build")
        if installation is not None:
            self.using_installed = True
            self._configure_library(*installation)
            tools.progress(f"Found NCCL: headers in {installation[0]}, library in {installation[1]}; skipping NCCL source build")
        else:
            tools.progress("No NCCL development installation found; building NCCL from source")
            self._build_nccl()
        try:
            self._build_tests(force=not self.using_installed)
        except (runtime.CommandError, runtime.OutputError) as exc:
            self._rebuild_after_failure(exc)

    def _configure_library(self, include, library):
        self.library_path = library / "libnccl.so"
        self.env["NCCL_HOME"] = str(include.parent)
        self.env["LD_LIBRARY_PATH"] = os.pathsep.join(
            value for value in (str(library), os.environ.get("LD_LIBRARY_PATH")) if value
        )
        for variable, flag in (("NVCUFLAGS", f"-I{shlex.quote(str(include))}"),
                               ("NVLDFLAGS", f"-L{shlex.quote(str(library))}")):
            self.env[variable] = " ".join(value for value in (flag, os.environ.get(variable)) if value)

    def _build_nccl(self):
        self.using_installed = False
        source = tools.get_output_dir() / "nccl-source"
        tools.progress(f"Building a fresh NCCL library in {source}")
        tools.run_command(["git", "clone", "https://github.com/NVIDIA/nccl.git", str(source)])
        tools.run_command(["make", "-j", "src.build"], cwd=source)
        self._configure_library(source / "build" / "include", source / "build" / "lib")

    def _build_tests(self, *, force=False):
        self._report_version()
        source = self.root / "nccl-tests"
        if not source.is_dir():
            tools.run_command(["git", "clone", "https://github.com/NVIDIA/nccl-tests.git", str(source)])
        self.test_executable = source / "build" / "all_reduce_perf"
        if self.test_executable.is_file() and not force:
            tools.progress(f"NCCL test build skipped; reusing existing binary: {self.test_executable}")
            return
        tools.progress(f"Building NCCL AllReduce test against {self.env['NCCL_HOME']}")
        tools.run_command(
            ["make", "-B"] if force else ["make"],
            cwd=source, env=self.env, check=True,
        )

    def _report_version(self):
        result = tools.run_command(
            [sys.executable, "-c", _VERSION_QUERY, str(self.library_path)],
            env=self.env, capture_output=True, text=True,
        )
        try:
            version = int(result.stdout.strip())
        except ValueError as exc:
            raise runtime.OutputError(f"Invalid NCCL version from {self.library_path}: {result.stdout!r}") from exc
        if version <= 0:
            raise runtime.OutputError(f"Invalid NCCL version from {self.library_path}: {version}")
        divisor = 10000 if version >= 10000 else 1000
        tools.progress(f"Using NCCL {version // divisor}.{version % divisor // 100}.{version % 100} ({self.library_path})")

    def _rebuild_after_failure(self, error):
        if not self.using_installed:
            raise error
        tools.logger.error("NCCL tests failed with the detected library: %s", error)
        if not sys.stdin.isatty():
            raise runtime.CommandError(
                "NCCL tests failed with the detected library. A source rebuild requires approval; "
                "rerun in an interactive terminal to approve rebuilding and retrying."
            ) from error
        print("Build NCCL from scratch and retry NCCL tests? [y/N] ", end="", file=sys.stderr, flush=True)
        if sys.stdin.readline().strip().lower() not in ("y", "yes"):
            raise runtime.CommandError("NCCL source rebuild declined; tests were not retried.") from error
        tools.progress("NCCL source rebuild approved; rebuilding the library and tests")
        self._build_nccl()
        self._build_tests(force=True)

    def run(self):
        try:
            self._run()
        except (runtime.CommandError, runtime.OutputError) as exc:
            self._rebuild_after_failure(exc)
            self._run()

    def _run(self):
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        num_gpus = str(tools.run_command("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')).strip()
        if num_gpus == '4':
            self.algo = "Ring"
        tools.progress("Running NCCL AllReduce on " + num_gpus + " GPUs")

        env = {**self.env, "NCCL_ALGO": self.algo}
        results = tools.run_command(
            [str(self.test_executable), '-b', '8', '-e', '8G', '-f', '2', '-g', num_gpus, '-n', '40'],
            env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        res = results.stdout.decode('utf-8').split('\n')
        log = []
        for line in res:
            line = line.split()
            if len(line) == 13 and line[2] == "float":
                log.append(line[11])

        buffer.append(log)
        if len(log) != len(buffer[0]):
            raise tools.runtime.OutputError(f"Expected {len(buffer[0])} NCCL measurements, received {len(log)}")
        table1 = PrettyTable()
        runs = ["Message Size", "Bandwidth (" + self.algo + ")"]
        for i in range(len(buffer)):
            table1.add_column(runs[i], buffer[i])
        tools.detail(table1)
        tools.export_markdown("NCCL Bandwidth", f"The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce ({self.algo} algorithm) tests in-place operations, varying from 8 bytes to 8GB of data.", table1)
