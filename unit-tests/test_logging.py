from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from Infra import cli, runtime, tools


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.stderr = io.StringIO()
        with redirect_stderr(self.stderr):
            runtime.configure_logging(self.directory.name)
        self.addCleanup(runtime.close_logging)

    def test_streams_both_channels_and_preserves_stdout(self):
        with runtime.benchmark_context("sample"), runtime.stage("measurement"):
            result = runtime.run_command(
                [sys.executable, "-c", "import sys; print('result'); print('diagnostic', file=sys.stderr)"],
                capture_output=True, text=True,
            )
        self.assertEqual(result.stdout, "result\n")
        self.assertEqual(result.stderr, "diagnostic\n")
        self.assertNotIn("diagnostic", self.stderr.getvalue())
        log = (Path(self.directory.name) / "sample.log").read_text()
        self.assertIn("diagnostic", log)
        self.assertIn("stage=measurement", log)
        self.assertIn("exit_code=0", log)

    def test_build_stage_announces_check_without_changing_error_stage(self):
        with self.assertRaises(runtime.CommandError) as error:
            with runtime.benchmark_context("nccl"), runtime.stage("build"):
                raise runtime.CommandError("compiler failed")
        self.assertIn("INFO nccl: checking build requirements...", self.stderr.getvalue())
        self.assertNotIn("INFO nccl: build...", self.stderr.getvalue())
        self.assertEqual(error.exception.stage, "build")
        self.assertIn("stage=build", (Path(self.directory.name) / "nccl.log").read_text())

    def test_errors_timeouts_and_missing_executables(self):
        with self.assertRaisesRegex(runtime.CommandError, "specific failure"):
            runtime.run_command([sys.executable, "-c",
                                 "import sys; print('specific failure', file=sys.stderr); sys.exit(7)"])
        with self.assertRaises(runtime.CommandTimeoutError):
            runtime.run_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)
        with self.assertRaisesRegex(runtime.PrerequisiteError, "Executable not found"):
            runtime.run_command(["/no/such/benchmark-executable"])
        with self.assertRaises(runtime.CommandError):
            runtime.run_command("false | cat", shell=True)

    def test_redacts_commands_output_and_tracebacks(self):
        secret = "test-private-value-12345"
        with patch.dict(os.environ, {"HF_TOKEN": secret}), runtime.benchmark_context("sample"):
            runtime.run_command([sys.executable, "-c", f"print('{secret}')"])
            try:
                raise ValueError(secret)
            except ValueError:
                runtime.logger.exception("failure")
        for file in Path(self.directory.name).glob("*.log"):
            content = file.read_text()
            self.assertNotIn(secret, content)
            self.assertIn("[REDACTED]", content)

    def test_cleanup_preserves_primary_error_and_runs_on_interrupt(self):
        container = Mock()
        benchmark = Mock(container=container)
        container.kill.side_effect = RuntimeError("cleanup problem")
        with self.assertRaisesRegex(runtime.OutputError, "primary"):
            with runtime.managed_container(benchmark):
                raise runtime.OutputError("primary")
        self.assertIn("cleanup problem", (Path(self.directory.name) / "run.log").read_text())
        container.kill.side_effect = None
        benchmark.container = container
        with self.assertRaises(KeyboardInterrupt):
            with runtime.managed_container(benchmark):
                raise KeyboardInterrupt
        self.assertEqual(container.kill.call_count, 2)


    def test_output_is_logged_before_the_command_exits(self):
        marker = Path(self.directory.name) / "streamed"
        original = runtime.logger.debug
        def observe(message, *args, **kwargs):
            if message == "%s: %s" and args[0] == "stdout" and "STREAMREADY" in args[1]:
                marker.touch()
            original(message, *args, **kwargs)
        script = ("import time; from pathlib import Path; print('STREAMREADY', flush=True); "
                  "deadline = time.monotonic() + 1\n"
                  f"while not Path({str(marker)!r}).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
                  f"assert Path({str(marker)!r}).exists(), 'output was not streamed'")
        with patch.object(runtime.logger, "debug", side_effect=observe):
            runtime.run_command([sys.executable, "-c", script], timeout=2)

    def test_interrupt_terminates_the_owned_command(self):
        processes = []
        popen = runtime.subprocess.Popen
        def launch(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(runtime.subprocess, "Popen", side_effect=launch), \
                patch.object(runtime.queue.Queue, "get", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runtime.run_command([sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertIsNotNone(processes[0].poll())


    def test_configuration_and_measurements_fail_explicitly(self):
        path = Path(self.directory.name) / "config.json"
        path.write_text("not json")
        with self.assertRaises(runtime.ConfigurationError):
            runtime.load_config(path, "LLMBenchmark")
        path.write_text(json.dumps({"LLMBenchmark": {"models": {"example": {
            "type": "nvidia", "use_model": True, "input_sizes": [128], "output_sizes": []
        }}}}))
        with self.assertRaisesRegex(runtime.ConfigurationError, "equal length"):
            tools.load_model_config(path, "nvidia")
        with self.assertRaises(runtime.OutputError):
            tools.parse_stream_measurements("no measurements")
        measurements = "\n".join(f"{operation} 100.5" for operation in ("Copy", "Mul", "Add", "Triad", "Dot"))
        self.assertEqual(len(tools.parse_stream_measurements(measurements)), 5)
        with self.assertRaises(runtime.OutputError):
            tools.parse_stream_measurements(measurements.replace("Copy 100.5", "Copy nan"))


    def test_custom_environment_secrets_and_authorization_are_redacted(self):
        secret = "custom-command-secret-12345"
        runtime.run_command([sys.executable, "-c", "import os; print(os.environ['CUSTOM_TOKEN'])"],
                            env={**os.environ, "CUSTOM_TOKEN": secret})
        self.assertNotIn(secret, (Path(self.directory.name) / "run.log").read_text())
        self.assertNotIn("secret-value", runtime.redact("Authorization: Bearer secret-value"))

    def test_temporary_cleanup_does_not_replace_the_original_failure(self):
        directory = Mock(name="temporary")
        directory.name = "owned-directory"
        directory.cleanup.side_effect = OSError("cleanup denied")
        with patch.object(runtime.tempfile, "TemporaryDirectory", return_value=directory):
            with self.assertRaisesRegex(runtime.OutputError, "primary failure"):
                with runtime.temporary_directory():
                    raise runtime.OutputError("primary failure")
        self.assertIn("cleanup denied", self.stderr.getvalue())
        self.assertNotIn("Traceback", self.stderr.getvalue())


    def test_container_output_mount_and_failed_start_cleanup(self):
        output = str(Path(self.directory.name).resolve())
        self.assertIn(output, tools.container_volumes("/project"))
        with patch.object(runtime, "run_command", return_value=SimpleNamespace(stdout="result")) as command:
            result = runtime.run_container_command(Mock(id="owned"), ["python", "benchmark.py"])
            self.assertEqual(result, "result")
            self.assertIn(f"BENCHMARK_OUTPUT_DIR={output}", command.call_args.args[0])
        class DockerFailure(Exception):
            pass
        client = Mock()
        client.containers.run.side_effect = DockerFailure("startup failed")
        docker = SimpleNamespace(from_env=Mock(return_value=client))
        class ImageNotFound(DockerFailure):
            pass
        errors = SimpleNamespace(DockerException=DockerFailure, ImageNotFound=ImageNotFound)
        with patch.dict(sys.modules, {"docker": docker, "docker.errors": errors}), \
                patch.object(runtime, "run_command", return_value=SimpleNamespace(stdout="id")) as command:
            with self.assertRaisesRegex(runtime.CommandError, "startup failed"):
                runtime.start_container("test-image")
        name = client.containers.run.call_args.kwargs["name"]
        self.assertTrue(name.startswith("ai-benchmark-"))
        self.assertTrue(client.containers.run.call_args.kwargs["auto_remove"])
        self.assertIn(["docker", "rm", "--force", name], [call.args[0] for call in command.call_args_list])

    def test_container_start_pulls_only_missing_images(self):
        class DockerFailure(Exception):
            pass

        class ImageNotFound(DockerFailure):
            pass

        errors = SimpleNamespace(DockerException=DockerFailure, ImageNotFound=ImageNotFound)
        for missing in (False, True):
            with self.subTest(missing=missing):
                client = Mock()
                if missing:
                    client.images.get.side_effect = ImageNotFound("missing")
                docker = SimpleNamespace(from_env=Mock(return_value=client))
                with patch.dict(sys.modules, {"docker": docker, "docker.errors": errors}), \
                        patch.object(runtime, "run_command") as command:
                    result = runtime.start_container("test-image", detach=True)
                client.images.get.assert_called_once_with("test-image")
                self.assertIs(result, client.containers.run.return_value)
                self.assertTrue(client.containers.run.call_args.kwargs["detach"])
                if missing:
                    command.assert_called_once_with(["docker", "pull", "test-image"])
                else:
                    command.assert_not_called()
        self.assertIn("Using local Docker image test-image", self.stderr.getvalue())
        self.assertIn("not available locally; pulling", self.stderr.getvalue())

    def test_container_image_lookup_failure_does_not_pull_or_start(self):
        class DockerFailure(Exception):
            pass

        class ImageNotFound(DockerFailure):
            pass

        client = Mock()
        client.images.get.side_effect = DockerFailure("daemon unavailable")
        docker = SimpleNamespace(from_env=Mock(return_value=client))
        errors = SimpleNamespace(DockerException=DockerFailure, ImageNotFound=ImageNotFound)
        with patch.dict(sys.modules, {"docker": docker, "docker.errors": errors}), \
                patch.object(runtime, "run_command") as command:
            with self.assertRaisesRegex(runtime.CommandError, "daemon unavailable"):
                runtime.start_container("test-image")
        command.assert_not_called()
        client.containers.run.assert_not_called()

    def test_container_pull_failure_does_not_start(self):
        class DockerFailure(Exception):
            pass

        class ImageNotFound(DockerFailure):
            pass

        client = Mock()
        client.images.get.side_effect = ImageNotFound("missing")
        docker = SimpleNamespace(from_env=Mock(return_value=client))
        errors = SimpleNamespace(DockerException=DockerFailure, ImageNotFound=ImageNotFound)
        with patch.dict(sys.modules, {"docker": docker, "docker.errors": errors}), \
                patch.object(runtime, "run_command",
                             side_effect=runtime.CommandError("pull failed")) as command:
            with self.assertRaisesRegex(runtime.CommandError, "pull failed"):
                runtime.start_container("test-image")
        command.assert_called_once_with(["docker", "pull", "test-image"])
        client.containers.run.assert_not_called()


    def test_multichase_script_propagates_failed_probes(self):
        binary_dir = Path(self.directory.name) / "bin"
        binary_dir.mkdir()
        probe = binary_dir / "numactl"
        probe.write_text("#!/bin/sh\nexit 7\n")
        probe.chmod(0o700)
        script = Path(__file__).resolve().parents[1] / "Benchmarks/NVIDIA/run_multichase.sh"
        with self.assertRaisesRegex(runtime.CommandError, "exited with code 7"):
            runtime.run_command(["bash", str(script)],
                                env={**os.environ, "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"]})


    def test_output_directory_context_restores_after_errors(self):
        parent = Path(self.directory.name)
        child = parent / "hbm_run_1_test"
        with self.assertRaisesRegex(ValueError, "measurement failed"):
            with runtime.benchmark_context("hbm", child):
                self.assertEqual(tools.get_output_dir(), child)
                raise ValueError("measurement failed")
        self.assertEqual(tools.get_output_dir(), parent)
        with patch("Infra.runtime.logging.FileHandler", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                with runtime.benchmark_context("nccl", parent / "nccl_run_1_test"):
                    self.fail("An unavailable log must not start the benchmark")
        self.assertEqual(tools.get_output_dir(), parent)



class CLITests(unittest.TestCase):
    def run_cli(self, directory, arguments, execute, discover=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main("NVIDIA", [*arguments, "--output-dir", directory],
                            execute=execute, discover=discover or (lambda: "NVIDIA_H200"))
        status_paths = list(Path(directory).glob("*/status.json"))
        status_path = next((path for path in status_paths if path.parent.name.startswith("suite_run_")), status_paths[0])
        return code, stdout.getvalue(), stderr.getvalue(), json.loads(status_path.read_text()), status_path.parent

    def test_stop_on_error_and_continue_on_error(self):
        for flags, expected_calls, expected_status in (
            ([], 1, ["FAIL", "SKIPPED"]),
            (["--continue-on-error"], 2, ["FAIL", "PASS"]),
        ):
            with self.subTest(flags=flags), TemporaryDirectory() as directory:
                execute = Mock(side_effect=[runtime.CommandError("build failed"), None])
                code, stdout, stderr, status, run_dir = self.run_cli(
                    directory, ["hbm", "nccl", *flags], execute)
                self.assertEqual(code, 1)
                self.assertEqual(execute.call_count, expected_calls)
                self.assertEqual([row["status"] for row in status["benchmarks"]], expected_status)
                self.assertIn("build failed", stderr)
                self.assertNotIn("build failed", stdout)
                self.assertFalse((run_dir / "summary.md").exists())
                benchmark_dir = Path(status["benchmarks"][0]["output_dir"])
                self.assertIn("Failure details", (benchmark_dir / "hbm.log").read_text())

    def test_verbosity_modes_and_results_location(self):
        def execute(name, machine):
            tools.progress("user progress")
            tools.detail("diagnostic detail")
            tools.export_markdown("sample", "result=42")
        for flags, progress, detail in (([], True, False), (["--quiet"], False, False),
                                       (["--verbose"], True, True)):
            with self.subTest(flags=flags), TemporaryDirectory() as directory:
                code, stdout, stderr, status, run_dir = self.run_cli(directory, ["hbm", *flags], execute)
                self.assertEqual(code, 0)
                self.assertEqual("user progress" in stderr, progress)
                self.assertEqual("diagnostic detail" in stderr, detail)
                self.assertIn("PASSED", stdout)
                self.assertEqual(stdout.count("result=42"), 1)
                self.assertLess(stdout.index("result=42"), stdout.index("Overall: PASSED"))
                self.assertIn("result=42", (run_dir / "summary.md").read_text())
                self.assertNotIn("## Run status", (run_dir / "summary.md").read_text())
                self.assertNotIn("Overall:", (run_dir / "summary.md").read_text())
                self.assertIn("result=42", (run_dir / "run.log").read_text())
                self.assertIn("result=42", (run_dir / "hbm.log").read_text())
                self.assertIn("diagnostic detail", (run_dir / "run.log").read_text())

    def test_result_tables_and_text_are_printed_and_saved_for_each_benchmark(self):
        rendered_table = (
            "+-----------+-------+\n"
            "| Operation |  GB/s |\n"
            "+-----------+-------+\n"
            "|    Copy   | 123.5 |\n"
            "+-----------+-------+"
        )

        def execute(name, machine):
            table = MagicMock(field_names=["Operation", "GB/s"], rows=[["Copy", 123.5]])
            table.__str__.return_value = rendered_table
            tools.export_markdown(f"{name} results", "Bandwidth measurements", table)
            tools.export_markdown(None, f"{name} raw result=42")

        with TemporaryDirectory() as directory:
            code, stdout, stderr, status, run_dir = self.run_cli(
                directory, ["hbm", "nccl"], execute)
            self.assertEqual(code, 0)
            self.assertEqual(stdout.count(rendered_table), 2)
            self.assertNotIn("Bandwidth measurements", stdout)
            self.assertNotIn("| --- | --- |", stdout)
            for row in status["benchmarks"]:
                name = row["benchmark"]
                self.assertNotIn(f"{name} results", stdout)
                self.assertEqual(stdout.count(f"{name} raw result=42"), 1)
                self.assertNotIn(f"{name} raw result=42", stderr)
                summary = (Path(row["output_dir"]) / "summary.md").read_text()
                self.assertIn(f"## {name} results", summary)
                self.assertIn("Bandwidth measurements", summary)
                self.assertIn("| --- | --- |", summary)
                self.assertIn("| Copy | 123.5 |", summary)
                self.assertNotIn(rendered_table, summary)
                for path in (Path(row["output_dir"]) / f"{name}.log", run_dir / "run.log"):
                    self.assertIn(f"{name} raw result=42", path.read_text())
                    self.assertIn(f"{name} results", path.read_text())
                    self.assertIn("Bandwidth measurements", path.read_text())
                    self.assertIn(rendered_table, path.read_text())

    def test_printed_results_redact_credentials(self):
        secret = "private-result-token-12345"

        def execute(name, machine):
            tools.export_markdown("sample", f"Token: {secret}")

        with TemporaryDirectory() as directory, patch.dict(os.environ, {"HF_TOKEN": secret}):
            code, stdout, stderr, status, run_dir = self.run_cli(directory, ["hbm"], execute)
            self.assertEqual(code, 0)
            self.assertNotIn(secret, stdout)
            self.assertIn("Token: [REDACTED]", stdout)
            self.assertNotIn(secret, (run_dir / "run.log").read_text())

    def test_interrupt_skip_and_setup_failure(self):
        for error, code, status in (
            (KeyboardInterrupt(), 130, "INTERRUPTED"),
            (runtime.UnsupportedHardwareError("unsupported GPU"), 0, "SKIPPED"),
        ):
            with self.subTest(error=error), TemporaryDirectory() as directory:
                execute = Mock(side_effect=error)
                result = self.run_cli(directory, ["hbm", "nccl"], execute)
                self.assertEqual(result[0], code)
                self.assertEqual(result[3]["benchmarks"][0]["status"], status)
        with TemporaryDirectory() as directory:
            execute = Mock()
            result = self.run_cli(directory, ["hbm"], execute,
                                  Mock(side_effect=runtime.PrerequisiteError("driver missing")))
            self.assertEqual(result[0], 1)
            execute.assert_not_called()
            self.assertIn("driver missing", result[2])

    def test_help_and_invalid_names_do_not_discover_hardware(self):
        for arguments, expected in ((["--help"], 0), (["not-a-benchmark"], 2)):
            with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                discover = Mock()
                with self.assertRaises(SystemExit) as error:
                    cli.main("AMD", arguments, discover=discover)
                self.assertEqual(error.exception.code, expected)
                discover.assert_not_called()


    def test_timeout_option_and_unexpected_errors(self):
        for execute, arguments, message in (
            (lambda name, machine: runtime.run_command([sys.executable, "-c", "import time; time.sleep(30)"]),
             ["hbm", "--timeout", "0.05"], "exceeded"),
            (Mock(side_effect=TypeError("implementation bug")), ["hbm"], "internal error"),
        ):
            with self.subTest(arguments=arguments), TemporaryDirectory() as directory:
                code, stdout, stderr, status, run_dir = self.run_cli(directory, arguments, execute)
                self.assertEqual(code, 1)
                self.assertIn(message, stderr)
                self.assertNotIn("Traceback", stderr)
                self.assertIn("Traceback", (run_dir / "hbm.log").read_text())
                self.assertEqual(status["status"], "FAILED")


    def test_markdown_is_published_only_after_successful_completion(self):
        def execute(name, machine):
            summary = tools.get_output_dir() / "summary.md"
            for description in ("first measurement", "second measurement"):
                tools.export_markdown("sample", description)
                self.assertFalse(summary.exists())

        with TemporaryDirectory() as directory:
            code, stdout, stderr, status, run_dir = self.run_cli(directory, ["hbm"], execute)
            self.assertEqual(code, 0)
            self.assertEqual((run_dir / "summary.md").read_text(),
                             "## sample\n\nfirst measurement\n\n\n## sample\n\nsecond measurement\n\n\n")
            self.assertIn("Overall: PASSED", stdout)

    def test_unsuccessful_benchmarks_do_not_export_partial_markdown(self):
        for error, expected_code, expected_status in (
            (runtime.CommandError("measurement failed"), 1, "FAIL"),
            (runtime.UnsupportedHardwareError("unsupported GPU"), 0, "SKIPPED"),
            (KeyboardInterrupt(), 130, "INTERRUPTED"),
            (TypeError("implementation bug"), 1, "FAIL"),
        ):
            with self.subTest(status=expected_status, error=error), TemporaryDirectory() as directory:
                def execute(name, machine):
                    tools.export_markdown("partial", "partial result=42")
                    raise error

                code, stdout, stderr, status, run_dir = self.run_cli(directory, ["hbm"], execute)
                self.assertEqual(code, expected_code)
                self.assertEqual(status["benchmarks"][0]["status"], expected_status)
                self.assertEqual(list(Path(directory).rglob("*.md")), [])
                self.assertIn("partial result=42", stdout)
                self.assertIn("partial result=42", (run_dir / "hbm.log").read_text())

    def test_mixed_suite_exports_only_passing_measurements(self):
        def execute(name, machine):
            tools.export_markdown(name, f"{name} measurement")
            if name == "nccl":
                raise runtime.OutputError("incomplete measurements")

        with TemporaryDirectory() as directory:
            code, stdout, stderr, status, run_dir = self.run_cli(
                directory, ["hbm", "nccl", "fa", "--continue-on-error"], execute)
            self.assertEqual(code, 1)
            self.assertEqual([row["status"] for row in status["benchmarks"]], ["PASS", "FAIL", "PASS"])
            self.assertFalse((run_dir / "summary.md").exists())
            for row in status["benchmarks"]:
                summary = Path(row["output_dir"]) / "summary.md"
                self.assertEqual(summary.exists(), row["status"] == "PASS")
                if summary.exists():
                    self.assertNotIn("Run status", summary.read_text())
                    self.assertNotIn("Overall:", summary.read_text())

    def test_markdown_publication_failure_does_not_leave_an_export_or_report_success(self):
        def execute(name, machine):
            tools.export_markdown("sample", "result=42")

        with TemporaryDirectory() as directory, patch.object(Path, "replace", side_effect=OSError("disk full")):
            code, stdout, stderr, status, run_dir = self.run_cli(directory, ["hbm"], execute)
            self.assertEqual(code, 1)
            self.assertEqual(status["benchmarks"][0]["status"], "FAIL")
            self.assertIn("disk full", stderr)
            self.assertEqual(list(Path(directory).rglob("*.md")), [])
            self.assertIn("Overall: FAILED", stdout)

    def test_status_write_failure_is_not_reported_as_success(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stdout, \
                redirect_stderr(io.StringIO()), patch.object(Path, "write_text", side_effect=OSError("disk full")):
            code = cli.main("NVIDIA", ["hbm", "--output-dir", directory],
                            execute=Mock(), discover=lambda: "NVIDIA_H200")
        self.assertEqual(code, 1)
        self.assertIn("Overall: FAILED", stdout.getvalue())
        self.assertNotIn("Overall: PASSED", stdout.getvalue())


class RunDirectoryTests(unittest.TestCase):
    def test_numbering_uses_highest_existing_directory_and_independent_counters(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            for name in ("hbm_run_2_old", "hbm_run_10_old", "nccl_run_7_old", "legacy-run"):
                (parent / name).mkdir()
            (parent / "hbm_run_999_old").write_text("not a run directory")
            first = cli.reserve_run_directory(parent, "hbm", "timestamp")
            second = cli.reserve_run_directory(parent, "hbm", "timestamp")
            other = cli.reserve_run_directory(parent, "nccl", "timestamp")
            self.assertEqual(first.name, "hbm_run_11_timestamp")
            self.assertEqual(second.name, "hbm_run_12_timestamp")
            self.assertEqual(other.name, "nccl_run_8_timestamp")
            self.assertTrue((parent / "hbm_run_2_old").is_dir())

    def test_concurrent_allocations_are_unique_and_sequential(self):
        with TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(
                lambda _: cli.reserve_run_directory(directory, "hbm", "timestamp"), range(16)
            ))
            self.assertEqual({path.name for path in paths}, {
                f"hbm_run_{number}_timestamp" for number in range(1, 17)
            })

    def test_repeated_cli_runs_use_benchmark_names_and_keep_previous_results(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            seen = []

            def execute(name, machine):
                output = tools.get_output_dir()
                seen.append(output)
                (output / "measurement.txt").write_text(f"result {len(seen)}")

            for name in ("hbm", "hbm", "nccl"):
                self.assertEqual(cli.main("NVIDIA", [name, "--output-dir", directory],
                                          execute=execute, discover=lambda: "NVIDIA_H200"), 0)
            for path, prefix in zip(seen, ("hbm_run_1_", "hbm_run_2_", "nccl_run_1_")):
                self.assertEqual(path.parent, Path(directory))
                self.assertRegex(path.name, rf"^{prefix}\d{{8}}T\d{{6}}Z$")
                self.assertTrue((path / "run.log").is_file())
                self.assertTrue((path / "status.json").is_file())
            self.assertEqual((seen[0] / "measurement.txt").read_text(), "result 1")

    def test_failure_interruption_and_continuation_keep_correct_folder_statuses(self):
        cases = (
            (runtime.CommandError("measurement failed"), [], 1, "FAIL", "SKIPPED"),
            (KeyboardInterrupt(), [], 130, "INTERRUPTED", "SKIPPED"),
            (runtime.CommandError("measurement failed"), ["--continue-on-error"], 1, "FAIL", "PASS"),
        )
        for error, flags, expected_code, first_status, second_status in cases:
            with self.subTest(status=first_status, flags=flags), TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                def execute(name, machine):
                    if name == "hbm":
                        raise error

                code = cli.main("NVIDIA", ["hbm", "nccl", *flags, "--output-dir", directory],
                                execute=execute, discover=lambda: "NVIDIA_H200")
                self.assertEqual(code, expected_code)
                suite_path = next(Path(directory).glob("suite_run_1_*/status.json"))
                suite = json.loads(suite_path.read_text())
                self.assertEqual(suite["exit_code"], expected_code)
                self.assertEqual([row["status"] for row in suite["benchmarks"]], [first_status, second_status])
                first, second = suite["benchmarks"]
                local = json.loads((Path(first["output_dir"]) / "status.json").read_text())
                self.assertEqual(local["exit_code"], expected_code)
                if second_status == "SKIPPED":
                    self.assertIsNone(second["output_dir"])
                    self.assertEqual(list(Path(directory).glob("nccl_run_*")), [])
                else:
                    local = json.loads((Path(second["output_dir"]) / "status.json").read_text())
                    self.assertEqual(local["exit_code"], 0)
                    self.assertEqual(local["status"], "PASSED")

    def test_multiple_and_all_benchmarks_route_results_and_status_to_separate_folders(self):
        for arguments in (["hbm", "nccl"], ["all"]):
            with self.subTest(arguments=arguments), TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                seen = {}

                def execute(name, machine):
                    seen[name] = tools.get_output_dir()
                    (seen[name] / "measurement.txt").write_text(name)

                code = cli.main("NVIDIA", [*arguments, "--output-dir", directory],
                                execute=execute, discover=lambda: "NVIDIA_H200")
                self.assertEqual(code, 0)
                suite_path = next(Path(directory).glob("suite_run_1_*/status.json"))
                suite = json.loads(suite_path.read_text())
                self.assertEqual(len(suite["benchmarks"]), len(seen))
                self.assertGreaterEqual(len(seen), 2)
                for row in suite["benchmarks"]:
                    name, path = row["benchmark"], Path(row["output_dir"])
                    self.assertEqual(path, seen[name])
                    self.assertEqual(path.parent, Path(directory))
                    self.assertTrue(path.name.startswith(f"{name}_run_1_"))
                    self.assertEqual((path / "measurement.txt").read_text(), name)
                    self.assertTrue((path / f"{name}.log").is_file())
                    local = json.loads((path / "status.json").read_text())
                    self.assertEqual(local["status"], "PASSED")
                    self.assertEqual(local["exit_code"], 0)
                    self.assertEqual(local["benchmarks"], [row])
                    self.assertFalse((suite_path.parent / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
