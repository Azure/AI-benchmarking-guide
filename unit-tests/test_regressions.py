from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from Infra import tools
from Infra import cli, runtime


ROOT = Path(__file__).resolve().parents[1]


class Table:
    def __init__(self, fields=None):
        self.field_names = fields or []
        self.rows = []

    def add_row(self, row):
        if len(row) != len(self.field_names):
            raise ValueError("Incorrect table row length")
        self.rows.append(row)

    def add_column(self, name, values):
        self.field_names.append(name)
        if not self.rows:
            self.rows = [[value] for value in values]
        else:
            if len(values) != len(self.rows):
                raise ValueError("Incorrect table column length")
            for row, value in zip(self.rows, values):
                row.append(value)


def load_module(relative_path):
    prettytable = ModuleType("prettytable")
    prettytable.PrettyTable = Table
    numpy = ModuleType("numpy")
    numpy.std = statistics.pstdev
    numpy.mean = statistics.mean
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = ModuleType("matplotlib.pyplot")
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = Mock()
    dependencies = {
        "prettytable": prettytable, "numpy": numpy,
        "docker": ModuleType("docker"), "huggingface_hub": hub,
        "matplotlib": matplotlib, "matplotlib.pyplot": matplotlib.pyplot,
    }
    spec = importlib.util.spec_from_file_location("module_under_test", ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


@contextmanager
def working_directory(path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.log = patch.object(tools, "write_log").start()
        self.export = patch.object(tools, "export_markdown").start()
        self.command = patch.object(tools, "run_command", return_value=SimpleNamespace(stdout=b"", stderr=b"")).start()
        self.container_command = patch.object(tools, "run_container_command", return_value="").start()
        patch.object(runtime, "run_command", return_value=SimpleNamespace(stdout="")).start()
        self.addCleanup(patch.stopall)

    def test_transferbench_accepts_ascii_and_unicode_rows(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        for separator in ("|", "\u2502"):
            with self.subTest(separator=separator):
                output = "=== TransferBench ===\n"
                for executor, bandwidth, total in (
                    (0, "55.506", "55.632"), (0, "56.556", "56.681"),
                ):
                    output += (
                        f"  Executor: GPU {executor:02d} {separator} {bandwidth} GB/s "
                        f"{separator} 4.836 ms {separator} 268435456 bytes "
                        f"{separator} {total} GB/s (sum)\n"
                    )
                matrix = (
                    "SRC\\DST,GPU 00,GPU 01,\n"
                    "GPU 00 ->,N/A,170.123,\n"
                    "GPU 00 <-,N/A,180.456,\n"
                    "GPU 00 <->,N/A,350.579,\n"
                    "GPU 01 <->,351.234,N/A,\n"
                )
                self.command.side_effect = [
                    SimpleNamespace(stdout=output.encode("utf-8")),
                    SimpleNamespace(stdout=matrix.encode("utf-8")),
                ]
                benchmark.run()
                table = self.export.call_args_list[-2].args[2]
                self.assertEqual(table.rows, [
                    ["Host to Device memcpy", "55.506 GB/s"],
                    ["Device to Host memcpy", "56.556 GB/s"],
                ])
                self.assertEqual(self.export.call_args.args[2].rows, [
                    ["GPU 0", "-", "350.579"], ["GPU 1", "351.234", "-"],
                ])
                self.assertEqual(self.command.call_args_list[-2].args[0], [
                    "sudo", str(ROOT / "TransferBench/build/TransferBench"),
                    str(ROOT / "Benchmarks/AMD/transferbench.cfg"),
                ])
                self.assertNotIn("shell", self.command.call_args.kwargs)

    def test_transferbench_rejects_missing_extra_or_invalid_measurements(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        valid = "Executor: GPU 00 | 55.506 GB/s | 4.836 ms | 55.632 GB/s (sum)\n"
        outputs = ["", valid, valid * 3, valid * 5, "invalid (sum)\n"]
        outputs += [
            valid.replace("55.506 GB/s", value) + valid
            for value in ("nan GB/s", "inf GB/s", "0 GB/s", "-1 GB/s", "bad GB/s", "55 MB/s")
        ]
        for output in outputs:
            with self.subTest(output=output):
                self.command.return_value.stdout = output.encode("utf-8")
                self.export.reset_mock()
                with self.assertRaises(runtime.OutputError):
                    benchmark.run()
                self.export.assert_not_called()

    def test_transferbench_rejects_unexpected_gpu_executors(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        row = "Executor: GPU 00 | 55.506 GB/s | 4.836 ms | 55.632 GB/s (sum)\n"
        for output in (row + row.replace("GPU 00", "GPU 01"), row.replace("GPU 00", "CPU 00") + row):
            with self.subTest(output=output):
                self.command.return_value.stdout = output.encode("utf-8")
                with self.assertRaises(runtime.OutputError):
                    benchmark.run()
                self.export.assert_not_called()

    def test_transferbench_config_keeps_host_tests_separate_from_gpu_matrix(self):
        config = (ROOT / "Benchmarks/AMD/transferbench.cfg").read_text()
        transfers = [line.strip() for line in config.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(transfers, [
            "1 32 (C0 G0 G0)", "1 32 (G0 G0 C0)",
        ])

    def test_transferbench_gpu_matrix_covers_all_eight_gpus_and_uses_total_rows(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        output = "SRC\\DST," + ",".join(f"GPU {gpu:02d}" for gpu in range(8)) + ",\n"
        for gpu in range(8):
            values = ["N/A" if peer == gpu else f"{100 * gpu + peer + 1:.2f}" for peer in range(8)]
            output += f"GPU {gpu:02d} ->," + ",".join(["1"] * 8) + ",\n"
            output += f"GPU {gpu:02d} <->," + ",".join(values) + ",\n"
        self.command.return_value.stdout = output.encode("utf-8")
        matrix = benchmark._run_gpu_matrix("/path with spaces/TransferBench")
        self.assertEqual(matrix.field_names, ["GPU", *[f"GPU {gpu}" for gpu in range(8)]])
        self.assertEqual(len(matrix.rows), 8)
        for gpu, row in enumerate(matrix.rows):
            self.assertEqual(row, [f"GPU {gpu}", *[
                "-" if peer == gpu else f"{100 * gpu + peer + 1:.2f}" for peer in range(8)
            ]])
        self.assertEqual(self.command.call_args.args[0], [
            "sudo", "env", "-u", "NUM_GPU_DEVICES", "P2P_MODE=2", "NUM_CPU_DEVICES=0",
            "NUM_GPU_SE=32", "USE_REMOTE_READ=1", "USE_GPU_DMA=0", "USE_GPU_TDM=0",
            "SHOW_ITERATIONS=0", "OUTPUT_TO_CSV=1", "/path with spaces/TransferBench", "p2p",
        ])

    def test_transferbench_gpu_matrix_rejects_bad_shapes_and_bandwidths(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        header = "SRC\\DST,GPU 00,GPU 01,\n"
        first = "GPU 00 <->,N/A,100.00,\n"
        second = "GPU 01 <->,101.00,N/A,\n"
        valid = header + first + second
        invalid = [
            "", first + second, header + first, valid + first, header + header + first + second,
            valid.replace("GPU 01,", "GPU 00,", 1),
            valid.replace("GPU 01 <->", "GPU 02 <->"),
            valid.replace("N/A,100.00,", "N/A,"),
            valid.replace("N/A,100.00,", "N/A,100.00,200.00,"),
            valid.replace("N/A,100.00,", "100.00,100.00,"),
            "SRC\\DST,GPU 00,\nGPU 00 <->,N/A,\n",
        ]
        invalid += [valid.replace("100.00", value) for value in ("nan", "inf", "0", "-1", "bad")]
        for output in invalid:
            with self.subTest(output=output):
                self.command.return_value.stdout = output.encode("utf-8")
                with self.assertRaises(runtime.OutputError):
                    benchmark._run_gpu_matrix("TransferBench")
                self.export.assert_not_called()

    def test_transferbench_gpu_matrix_marks_unavailable_peer_paths_with_warning(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        self.command.return_value.stdout = (
            b"SRC\\DST,GPU 00,GPU 01,\nGPU 00 <->,N/A,N/A,\nGPU 01 <->,100.00,N/A,\n"
        )
        with patch.object(tools.logger, "warning") as warning:
            matrix = benchmark._run_gpu_matrix("TransferBench")
        self.assertEqual(matrix.rows[0], ["GPU 0", "-", "-"])
        warning.assert_called_once_with(
            "TransferBench peer path unavailable: GPU %s <-> GPU %s", 0, 1,
        )

    def test_transferbench_propagates_command_failure(self):
        module = load_module("Benchmarks/AMD/TransferBench.py")
        benchmark = module.TransferBench("unused", str(ROOT), "MI300X")
        self.command.side_effect = runtime.CommandError("TransferBench failed")
        with self.assertRaises(runtime.CommandError):
            benchmark.run()
        self.export.assert_not_called()

    def test_amd_hbm_uses_fixed_runs_and_delay_without_config(self):
        module = load_module("Benchmarks/AMD/HBMBandwidth.py")
        with patch.object(tools, "load_config", side_effect=AssertionError("HBM must not load config")):
            benchmark = module.HBMBandwidth(str(ROOT), "MI300X")
        self.assertEqual((benchmark.num_runs, benchmark.interval), (5, 10))
        self.assertEqual(benchmark.process_stats([1_000_000]), [1, 1, 1])
        self.assertNotIn("HBMBandwidth", json.loads((ROOT / "config.json").read_text()))
        measurements = "\n".join(f"{operation} 1000000" for operation in ("Copy", "Mul", "Add", "Triad", "Dot"))
        with patch.object(tools, "run_command",
                          return_value=SimpleNamespace(stdout=measurements.encode())) as command, \
                patch.object(module.time, "sleep") as sleep, patch.object(benchmark, "save_results") as save:
            benchmark.run()
        self.assertEqual(command.call_count, 5)
        self.assertEqual([call.args for call in sleep.call_args_list], [(10,)] * 5)
        self.assertEqual(len(benchmark.buffer), 5)
        save.assert_called_once()

    def test_amd_hbm_cli_dispatch_does_not_pass_configuration(self):
        module = SimpleNamespace(HBMBandwidth=Mock())
        with patch.object(cli, "import_module", return_value=module):
            cli.run_benchmark("AMD", "hbm", "MI300X")
        module.HBMBandwidth.assert_called_once_with(str(ROOT), "MI300X")
        module.HBMBandwidth.return_value.build.assert_called_once()
        module.HBMBandwidth.return_value.run.assert_called_once()

    def test_amd_container_settings_preserve_benchmark_specific_options(self):
        cases = (
            ("GEMMHipblasLt", "GEMMHipBLAS", "rocm/vllm-dev:main"),
            ("RCCLBandwidth", "RCCLBandwidth", "rocm/vllm-dev:main"),
            ("FlashAttention", "FlashAttention", "rocm/vllm-dev:main"),
            ("LLMBenchmark", "LLMBenchmark", "rocm/vllm-dev:20241121-tuned"),
        )
        for module_name, class_name, image in cases:
            with self.subTest(benchmark=class_name):
                module = load_module(f"Benchmarks/AMD/{module_name}.py")
                cls = getattr(module, class_name)
                benchmark = cls(str(ROOT), "MI300X") if class_name == "FlashAttention" else \
                    cls(str(ROOT / "config.json"), str(ROOT), "MI300X")
                with patch.object(tools, "get_output_dir", return_value=Path("/results")), \
                        patch.object(tools, "start_container") as start:
                    benchmark.create_container()
                self.assertEqual(start.call_args.args, (image,))
                options = start.call_args.kwargs
                self.assertEqual(options["devices"], ["/dev/kfd", "/dev/dri", "/dev/mem"])
                self.assertEqual(options["group_add"], ["render"])
                self.assertEqual(options["cap_add"], ["CAP_SYS_ADMIN", "SYS_PTRACE"])
                self.assertEqual(options["security_opt"], ["seccomp=unconfined"])
                self.assertEqual(options["ipc_mode"], "host")
                self.assertEqual(options["network"], "host")
                for flag in ("privileged", "tty", "detach", "auto_remove"):
                    self.assertTrue(options[flag])
                self.assertEqual(options["volumes"], {
                    str(ROOT): {"bind": str(ROOT), "mode": "rw"},
                    "/results": {"bind": "/results", "mode": "rw"},
                })
                if class_name == "FlashAttention":
                    self.assertNotIn("entrypoint", options)
                else:
                    self.assertEqual(options["entrypoint"], "/bin/bash")
                if class_name == "LLMBenchmark":
                    self.assertEqual(options["environment"], {"HF_HOME": str(ROOT)})
                else:
                    self.assertNotIn("environment", options)

    def test_amd_dispatcher_owns_one_cleanup_scope_for_all_container_stages(self):
        cases = (
            ("gemm", "GEMMHipBLAS", ["create_container", "build", "run_model_sizes"]),
            ("rccl", "RCCLBandwidth", ["create_container", "build", "run"]),
            ("fa", "FlashAttention", ["create_container", "run"]),
            ("llm", "LLMBenchmark", ["create_container", "run_benchmark"]),
        )
        for name, class_name, methods in cases:
            for failed_method in (None, *methods):
                for exception in (runtime.CommandError, KeyboardInterrupt):
                    with self.subTest(benchmark=name, failed_method=failed_method, exception=exception):
                        container = Mock()
                        instance = SimpleNamespace(container=None)
                        events = []

                        def execute(method):
                            events.append(method)
                            if method == "create_container":
                                instance.container = container
                            if method == failed_method:
                                raise exception("failed")

                        for method in methods:
                            setattr(instance, method, Mock(side_effect=lambda m=method: execute(m)))
                        module = SimpleNamespace(**{class_name: Mock(return_value=instance)})
                        with patch.object(cli, "import_module", return_value=module), \
                                patch.object(runtime, "managed_container",
                                             wraps=runtime.managed_container) as managed:
                            if failed_method is None:
                                cli.run_benchmark("AMD", name, "MI300X")
                            else:
                                with self.assertRaises(exception):
                                    cli.run_benchmark("AMD", name, "MI300X")
                        managed.assert_called_once_with(instance)
                        container.kill.assert_called_once()
                        self.assertIsNone(instance.container)
                        expected = methods if failed_method is None else methods[:methods.index(failed_method) + 1]
                        self.assertEqual(events, expected)

    def test_amd_container_options_are_independent_between_calls(self):
        with patch.object(tools, "get_output_dir", return_value=Path("/results")):
            first = tools.amd_container_options("/repo")
            first["devices"].append("/extra")
            first["volumes"].clear()
            second = tools.amd_container_options("/repo")
        self.assertNotIn("/extra", second["devices"])
        self.assertIn("/repo", second["volumes"])

    def test_rccl_build_uses_installed_library_and_only_pinned_allreduce_test(self):
        module = load_module("Benchmarks/AMD/RCCLBandwidth.py")
        output = Path("/results with spaces/rccl_run")
        with patch.object(tools, "get_output_dir", return_value=output):
            benchmark = module.RCCLBandwidth("unused", str(ROOT), "MI300X")
        self.container_command.return_value = "2.21.5\n"
        benchmark.build()
        commands = [call.args[1] for call in self.container_command.call_args_list]
        self.assertEqual(commands, [
            ["test", "-r", "/usr/local/include/rccl/rccl.h"],
            ["test", "-r", "/usr/local/lib/librccl.so"],
            ["python3", "-c", module._VERSION_QUERY],
            ["git", "clone", "https://github.com/ROCm/rccl-tests.git", str(output / "rccl-tests")],
            ["git", "-C", str(output / "rccl-tests"), "checkout", "--detach", module._TESTS_REVISION],
            ["make", "-C", str(output / "rccl-tests/src"), str(benchmark.test_executable),
             f"BUILDDIR={output / 'rccl-tests/build'}", "ROCM_PATH=/opt/rocm",
             "HIP_HOME=/opt/rocm", "NCCL_HOME=/usr/local", "CUSTOM_RCCL_LIB=/usr/local/lib"],
            ["test", "-x", str(benchmark.test_executable)],
        ])

    def test_rccl_build_stops_on_missing_headers_library_or_failed_compilation(self):
        module = load_module("Benchmarks/AMD/RCCLBandwidth.py")
        for failure_index in (0, 1, 2, 5, 6):
            with self.subTest(failure_index=failure_index):
                benchmark = module.RCCLBandwidth("unused", str(ROOT), "MI300X")
                self.container_command.reset_mock()
                self.container_command.side_effect = (
                    [""] * failure_index + [runtime.CommandError("dependency or build failed")]
                )
                with self.assertRaisesRegex(runtime.CommandError, "dependency or build failed"):
                    benchmark.build()
                self.assertEqual(self.container_command.call_count, failure_index + 1)
                self.export.assert_not_called()

    def test_rccl_run_uses_output_scoped_allreduce_binary(self):
        module = load_module("Benchmarks/AMD/RCCLBandwidth.py")
        output = Path("/results with spaces/rccl_run")
        with patch.object(tools, "get_output_dir", return_value=output):
            benchmark = module.RCCLBandwidth("unused", str(ROOT), "MI300X")
        self.container_command.return_value = "8 2 float sum -1 1 2 3 0 1 2 3 0\n" * 31
        benchmark.run()
        self.assertEqual(self.container_command.call_count, 4)
        for call, algorithm in zip(self.container_command.call_args_list, ("Tree", "Ring", "NVLS", "NVLSTree")):
            self.assertEqual(call.args[1][:3], [
                "env", f"NCCL_ALGO={algorithm}", str(output / "rccl-tests/build/all_reduce_perf"),
            ])
        table = self.export.call_args.args[2]
        self.assertEqual(len(table.rows), 31)
        self.assertEqual(table.field_names, ["Message Size", "Tree", "Ring", "NVLS", "NVLSTree"])

    def test_pretraining_machine_normalization(self):
        for name, expected in (("NVIDIA_H200", "H200"), ("NVIDIA_GB200", "GB200")):
            self.assertEqual(tools.get_pretraining_machine(name), expected)
        with self.assertRaises(ValueError):
            tools.get_pretraining_machine("NVIDIA_A100")

    def test_runner_rejects_unsupported_pretraining_gpu(self):
        with patch.object(cli, "import_module") as import_module:
            with self.assertRaises(runtime.UnsupportedHardwareError):
                cli.run_benchmark("NVIDIA", "llama_8b_pretrain", "NVIDIA_A100")
            import_module.assert_not_called()

    def test_steady_state_indices_and_boundaries(self):
        module = load_module("Benchmarks/NVIDIA/LLAMA3Run.py")
        self.assertEqual(module.detect_steady_state([2] * 12), (0, 2))
        self.assertEqual(module.detect_steady_state([2] * 11), (None, None))
        self.assertEqual(module.detect_steady_state([]), (None, None))
        self.assertEqual(module.detect_steady_state([100, 50, 25] + [2] * 12), (3, 2))
        self.assertEqual(module.detect_steady_state([0, 10] * 20), (None, None))

    def test_pretraining_command_logs_and_failure(self):
        module = load_module("Benchmarks/NVIDIA/LLAMA3Run.py")
        with TemporaryDirectory() as directory, working_directory(directory):
            Path("Outputs").mkdir()
            shared_log = Path("Outputs/log.txt")
            shared_log.write_text("previous benchmarks")
            benchmark = module.LLAMA3Pretraining(str(ROOT / "config.json"), "NVIDIA_H200")
            benchmark.plot_results = Mock(return_value=2)
            with patch.object(module.tools, "run_command") as run:
                benchmark.run()
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-v") + 1],
                                 f"{Path(directory).resolve()}:/workspace/nemo-run")
                self.assertEqual(command[command.index("-w") + 1], "/workspace/nemo-run")
                self.assertEqual(command[-2:], ["--machine_name", "H200"])
                self.assertEqual(command.count(benchmark.container), 1)
                self.assertTrue(run.call_args.kwargs["check"])
                benchmark.run()
            self.assertEqual(shared_log.read_text(), "previous benchmarks")
            self.assertEqual(len(list(Path("Outputs").glob("LLAMA3_8b_*.log"))), 2)
            benchmark.plot_results.reset_mock()
            self.export.reset_mock()
            with patch.object(module.tools, "run_command",
                              side_effect=subprocess.CalledProcessError(1, "docker")):
                with self.assertRaises(subprocess.CalledProcessError):
                    benchmark.run()
            benchmark.plot_results.assert_not_called()
            self.export.assert_not_called()

    def test_pretraining_rejects_empty_training_log(self):
        module = load_module("Benchmarks/NVIDIA/LLAMA3Run.py")
        benchmark = module.LLAMA3Pretraining(str(ROOT / "config.json"), "NVIDIA_H200")
        with patch("builtins.open", return_value=io.StringIO("training failed")):
            with self.assertRaisesRegex(ValueError, "No training measurements"):
                benchmark.plot_results("unused")

    def test_container_failure_is_logged_and_raised(self):
        container = Mock(id="test-container")
        with patch.object(runtime, "run_command", side_effect=runtime.CommandError("model failed")) as command:
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                runtime.run_container_command(container, ["python", "benchmark.py"])
        self.assertIn("test-container", command.call_args.args[0])

    def test_amd_llm_success_failure_and_missing_measurements(self):
        module = load_module("Benchmarks/AMD/LLMBenchmark.py")
        for code, output in (
            (0, b"Throughput: 2.00 requests/s, 512.00 total tokens/s, 256.25 output tokens/s\n"),
            (1, b"model failed"),
            (0, b"no measurements"),
        ):
            with self.subTest(code=code, output=output):
                benchmark = module.LLMBenchmark(str(ROOT / "config.json"), str(ROOT), "MI300X")
                container = benchmark.container = Mock()
                self.container_command.side_effect = runtime.CommandError(output.decode()) if code else None
                self.container_command.return_value = output.decode()
                self.export.reset_mock()
                if code:
                    with self.assertRaises(RuntimeError):
                        benchmark.run_benchmark()
                elif output == b"no measurements":
                    with self.assertRaises(ValueError):
                        benchmark.run_benchmark()
                else:
                    benchmark.run_benchmark()
                    self.assertEqual(len(benchmark.table.rows), 6)
                    self.assertEqual(benchmark.table.rows[0][-1], "256.25")
                container.kill.assert_not_called()
                if code or output == b"no measurements":
                    self.export.assert_not_called()

    def test_flash_attention_selects_only_requested_sequence_length(self):
        module = load_module("Benchmarks/AMD/FlashAttention.py")
        output = (
            "### causal=False, headdim=64, batch_size=2, seqlen=1024 ###\n"
            "Flash2 fwd + bwd: 999.0\nPytorch fwd + bwd: 888.0\n"
            "### causal=True, headdim=128, batch_size=2, seqlen=8192 ###\n"
            "Flash2 fwd + bwd: 123.5\nPytorch fwd + bwd: 45.25\n"
        )
        benchmark = module.FlashAttention("/repo with spaces", "MI300X")
        container = benchmark.container = Mock()
        self.container_command.return_value = output
        with patch.object(module.os, "chdir"), patch.object(module.os.path, "isdir", return_value=True), \
                patch.object(module.tools, "run_command"), patch.object(benchmark, "create_container"):
            benchmark.run()
        command = self.container_command.call_args.args[1]
        self.assertEqual(command, ["python3", "/repo with spaces/flash-attention/benchmarks/benchmark_flash_attention.py"])
        self.assertEqual(self.export.call_args.args[2].rows, [["True", 128, 123.5, 45.25]])
        container.kill.assert_not_called()

    def test_flash_attention_failure_leaves_cleanup_to_dispatcher_without_report(self):
        module = load_module("Benchmarks/AMD/FlashAttention.py")
        benchmark = module.FlashAttention(str(ROOT), "MI300X")
        container = benchmark.container = Mock()
        self.container_command.side_effect = runtime.CommandError("failed")
        with patch.object(module.os, "chdir"), patch.object(module.os.path, "isdir", return_value=True), \
                patch.object(module.tools, "run_command"), patch.object(benchmark, "create_container"):
            with self.assertRaises(RuntimeError):
                benchmark.run()
        container.kill.assert_not_called()
        self.export.assert_not_called()

    def test_fio_json_and_cleanup_preserve_unrelated_files(self):
        module = load_module("Infra/fio.py")
        output = json.dumps({"jobs": [{"error": 0, "read": {"bw_bytes": 2_097_152},
                                      "write": {"bw_bytes": 1_048_576}}]})
        result = SimpleNamespace(stdout=output, stderr="")
        with TemporaryDirectory() as directory, working_directory(directory):
            Path("Outputs").mkdir()
            unrelated = Path("Outputs/test-results.txt")
            unrelated.write_text("keep")
            with patch.object(module.tools, "run_command", return_value=result) as run:
                module.run_fio()
                self.assertEqual(run.call_count, 8)
                command = run.call_args.args[0]
                temporary_path = Path(next(arg.split("=", 1)[1] for arg in command
                                           if arg.startswith("--directory=")))
                self.assertFalse(temporary_path.exists())
                self.assertTrue(run.call_args.kwargs["check"])
            self.assertEqual(unrelated.read_text(), "keep")
            table = self.export.call_args.args[2]
            self.assertEqual(table.rows[0], ["read", "1M", 2])
            self.assertEqual(table.rows[3], ["write", "1M", 1])
            self.export.reset_mock()
            with patch.object(module.tools, "run_command",
                              side_effect=subprocess.CalledProcessError(1, "fio")):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.run_fio()
            self.assertEqual(list(Path("Outputs").iterdir()), [unrelated])
            self.export.assert_not_called()

    def test_fio_rejects_empty_results_and_job_errors(self):
        module = load_module("Infra/fio.py")
        for output, error in (
            ({"jobs": []}, ValueError),
            ({"jobs": [{"error": 5}]}, runtime.OutputError),
        ):
            with self.subTest(output=output), TemporaryDirectory() as directory, working_directory(directory):
                result = SimpleNamespace(stdout=json.dumps(output), stderr="")
                with patch.object(module.tools, "run_command", return_value=result):
                    with self.assertRaises(error):
                        module.run_fio()
                self.assertEqual(list(Path("Outputs").iterdir()), [])
                self.export.assert_not_called()

    def test_nccl_environment_reaches_build_and_run(self):
        module = load_module("Benchmarks/NVIDIA/NCCLBandwidth.py")
        benchmark = module.NCCLBandwidth("config.json", "NVIDIA_H200")
        result = SimpleNamespace(stdout=b"", stderr=b"")
        with patch.object(module, "find_nccl", return_value=(ROOT / "nccl/build/include", ROOT / "nccl/build/lib")), \
                patch.object(benchmark, "_report_version"), \
                patch.object(Path, "is_file", return_value=False), \
                patch.object(module.tools, "get_output_dir", return_value=Path("/isolated-output")), \
                patch.object(module.tools, "run_command", return_value=result) as run:
            benchmark.build()
        make = next(call for call in run.call_args_list if call.args[0][0] == "make")
        self.assertEqual(make.kwargs["env"]["NCCL_HOME"], str(ROOT / "nccl/build"))
        self.assertTrue(make.kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(ROOT / "nccl/build/lib")))
        row = "8 2 float sum -1 1 2 3 0 1 2 3 0\n"
        with patch.object(module.tools, "run_command", side_effect=[
            SimpleNamespace(stdout=b"8\n"), SimpleNamespace(stdout=(row * 31).encode(), stderr=b"")
        ]) as run:
            benchmark.run()
        self.assertEqual(run.call_args.kwargs["env"]["NCCL_ALGO"], "NVLS")
        self.assertEqual(run.call_args.kwargs["env"]["NCCL_HOME"], str(ROOT / "nccl/build"))
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_nccl_rejects_missing_results(self):
        module = load_module("Benchmarks/NVIDIA/NCCLBandwidth.py")
        benchmark = module.NCCLBandwidth("config.json", "NVIDIA_H200")
        with patch.object(module.tools, "run_command", side_effect=[
            SimpleNamespace(stdout=b"8\n"), SimpleNamespace(stdout=b"", stderr=b"")
        ]):
            with self.assertRaisesRegex(ValueError, "Expected 31 NCCL measurements"):
                benchmark.run()
        self.export.assert_not_called()

    def test_gemm_iteration_settings_reach_script(self):
        module = load_module("Benchmarks/AMD/GEMMHipblasLt.py")
        benchmark = module.GEMMHipBLAS("config.json", "/repo with spaces", "MI300X", i=123, w=456)
        container = benchmark.container = Mock()
        self.container_command.side_effect = runtime.CommandError("failed")
        with self.assertRaises(RuntimeError):
            benchmark.run_model_sizes()
        command = self.container_command.call_args.args[1]
        self.assertIn("cd '/repo with spaces/Benchmarks/AMD'", command[2])
        self.assertTrue(command[2].endswith(" 123 456"))
        container.kill.assert_not_called()
        script = (ROOT / "Benchmarks/AMD/hipBLAS_runner.sh").read_text()
        self.assertIn("iters: $iterations, cold_iters: $warmup_iterations", script)

    def test_gemm_shell_uses_requested_iterations_and_propagates_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            working = root / "Benchmarks/AMD"
            working.mkdir(parents=True)
            (root / "Outputs").mkdir()
            binary = root / "hipBLASLt/build/release/clients/staging/hipblaslt-bench"
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\ncat > "$GEMM_CAPTURE"\nprintf "T,N,0\\n"\n')
            binary.chmod(0o700)
            capture = root / "parameters.yaml"
            command = ["bash", str(ROOT / "Benchmarks/AMD/hipBLAS_runner.sh"),
                       "1024", "2048", "4096", "123", "456"]
            subprocess.run(command, cwd=working, env={**os.environ, "GEMM_CAPTURE": str(capture)},
                           check=True, capture_output=True)
            yaml = capture.read_text()
            self.assertIn("M: 1024, N: 2048, K: 4096", yaml)
            self.assertIn("iters: 123, cold_iters: 456", yaml)
            binary.write_text("#!/bin/sh\nexit 7\n")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(command, cwd=working, check=True, capture_output=True)

    def test_nvidia_throughput_preserves_fraction(self):
        module = load_module("Benchmarks/NVIDIA/LLMBenchmark.py")
        benchmark = module.LLMBenchmark.__new__(module.LLMBenchmark)
        benchmark.table = Table(["tp", "input", "output", "throughput"])
        output = ("TP Size: 1\nAverage Input Length (tokens): 128\n"
                  "Average Output Length (tokens): 128\nToken Throughput (tokens/sec): 1234.99\n")
        with patch("builtins.open", return_value=io.StringIO(output)):
            benchmark.extract_benchmark_info("unused")
        self.assertEqual(benchmark.table.rows, [["1", "128", "128", "1234.99"]])


if __name__ == "__main__":
    unittest.main()
