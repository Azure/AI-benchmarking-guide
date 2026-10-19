from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Infra import runtime, tools
from test_regressions import load_module


class NCCLTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.module = load_module("Benchmarks/NVIDIA/NCCLBandwidth.py")
        self.benchmark = self.module.NCCLBandwidth("config.json", "NVIDIA_H200")
        self.benchmark.root = self.root
        self.stderr = io.StringIO()
        with redirect_stderr(self.stderr):
            runtime.configure_logging(self.root / "output", quiet=True)
        self.addCleanup(runtime.close_logging)
        self.command = patch.object(tools, "run_command", side_effect=lambda command, **kwargs:
                                    SimpleNamespace(stdout="22809" if command[0] == sys.executable else b"",
                                                    stderr=b"")).start()
        self.addCleanup(patch.stopall)

    def installation(self, name="installed", library="lib"):
        prefix = self.root / name
        include, libraries = prefix / "include", prefix / library
        include.mkdir(parents=True)
        libraries.mkdir(parents=True)
        (include / "nccl.h").touch()
        (libraries / "libnccl.so").touch()
        return include, libraries

    def test_discovers_explicit_prefixes_and_multiarch_libraries(self):
        for variable, library in (("NCCL_HOME", "lib"), ("NCCL_ROOT", "lib64"),
                                  ("NCCL_HOME", "lib/aarch64-linux-gnu")):
            with self.subTest(variable=variable, library=library):
                include, libraries = self.installation(library.replace("/", "-"), library)
                found = self.module.find_nccl({variable: str(include.parent)}, self.root / "absent")
                self.assertEqual(found, (include, libraries))
        self.command.assert_not_called()

    def test_discovers_library_path_prefix(self):
        include, libraries = self.installation()
        found = self.module.find_nccl({"LD_LIBRARY_PATH": str(libraries)}, self.root / "absent")
        self.assertEqual(found, (include, libraries))

    def test_does_not_accept_missing_headers_or_unlinkable_libraries(self):
        include, libraries = self.installation()
        for files in ({include / "nccl.h"}, {libraries / "libnccl.so"},
                      {include / "nccl.h", libraries / "libnccl.so.2"}):
            with self.subTest(files=files), patch.object(
                Path, "is_file", autospec=True, side_effect=lambda path: path in files
            ):
                self.assertIsNone(self.module.find_nccl(
                    {"NCCL_HOME": str(include.parent)}, self.root / "incomplete"
                ))
        self.command.assert_not_called()

    def test_discovers_complete_local_build_when_no_system_installation_exists(self):
        include, libraries = self.installation("nccl/build")
        files = {include / "nccl.h", libraries / "libnccl.so"}
        with patch.object(Path, "is_file", autospec=True, side_effect=lambda path: path in files):
            found = self.module.find_nccl({}, include.parent)
        self.assertEqual(found, (include, libraries))

    def test_detected_library_skips_source_build_and_is_used_for_tests(self):
        include, libraries = self.installation(library="lib/aarch64-linux-gnu")
        (self.root / "nccl-tests").mkdir()
        with patch.object(self.module, "find_nccl", return_value=(include, libraries)), \
                patch.object(self.benchmark, "_build_nccl") as source_build:
            self.benchmark.build()
        source_build.assert_not_called()
        self.assertEqual(self.command.call_count, 2)
        command = self.command.call_args
        self.assertEqual(command.args[0][0], "make")
        self.assertEqual(command.kwargs["env"]["NCCL_HOME"], str(include.parent))
        self.assertTrue(command.kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(libraries)))
        self.assertIn(str(libraries), command.kwargs["env"]["NVLDFLAGS"])
        self.assertIn(str(include), command.kwargs["env"]["NVCUFLAGS"])
        self.assertEqual(command.args[0], ["make"])
        self.assertEqual(command.kwargs["cwd"], self.root / "nccl-tests")
        self.assertEqual(self.benchmark.test_executable, self.root / "nccl-tests/build/all_reduce_perf")

    def test_existing_test_binary_is_reused_without_compiling(self):
        include, libraries = self.installation()
        binary = self.root / "nccl-tests/build/all_reduce_perf"
        binary.parent.mkdir(parents=True)
        binary.write_text("existing executable")
        with patch.object(self.module, "find_nccl", return_value=(include, libraries)):
            self.benchmark.build()
            self.benchmark.build()
        self.assertEqual(self.command.call_count, 2)
        self.assertTrue(all(call.args[0][0] == sys.executable for call in self.command.call_args_list))
        self.assertEqual(self.benchmark.test_executable, binary)
        self.assertEqual(binary.read_text(), "existing executable")
        self.assertEqual(list((self.root / "output").glob("nccl-tests-*")), [])
        log = (self.root / "output/run.log").read_text()
        self.assertIn("NCCL test build skipped; reusing existing binary", log)
        self.assertNotIn("Building NCCL AllReduce test", log)

    def test_missing_test_checkout_is_cloned_once_into_repository(self):
        include, libraries = self.installation()
        with patch.object(self.module, "find_nccl", return_value=(include, libraries)):
            self.benchmark.build()
        self.assertEqual(self.command.call_args_list[1].args[0], [
            "git", "clone", "https://github.com/NVIDIA/nccl-tests.git", str(self.root / "nccl-tests")
        ])
        self.assertEqual(self.command.call_args_list[2].args[0], ["make"])
        self.assertEqual(self.command.call_args_list[2].kwargs["cwd"], self.root / "nccl-tests")

    def test_absent_installation_builds_fresh_source_without_touching_existing_tree(self):
        old_source = self.root / "nccl"
        old_source.mkdir()
        marker = old_source / "keep"
        marker.write_text("existing build")
        binary = self.root / "nccl-tests/build/all_reduce_perf"
        binary.parent.mkdir(parents=True)
        binary.touch()
        with patch.object(self.module, "find_nccl", return_value=None):
            self.benchmark.build()
        calls = self.command.call_args_list
        self.assertEqual(calls[0].args[0][:3], ["git", "clone", "https://github.com/NVIDIA/nccl.git"])
        self.assertEqual(calls[0].args[0][3], str(self.root / "output" / "nccl-source"))
        self.assertEqual(calls[1].kwargs["cwd"], self.root / "output" / "nccl-source")
        self.assertEqual(calls[3].args[0], ["make", "-B"])
        self.assertEqual(calls[3].kwargs["cwd"], self.root / "nccl-tests")
        self.assertEqual(marker.read_text(), "existing build")
        self.assertFalse(self.benchmark.using_installed)

    def test_failed_tests_request_approval_and_retry_once_after_rebuild(self):
        self.benchmark.using_installed = True
        binary = self.root / "nccl-tests/build/all_reduce_perf"
        binary.parent.mkdir(parents=True)
        binary.touch()
        failure = runtime.CommandError("installed library failed")
        with patch.object(self.benchmark, "_run", side_effect=[failure, None]) as run, \
                patch.object(self.module.sys, "stdin", io.StringIO("yes\n")) as stdin, \
                patch.object(stdin, "isatty", return_value=True), redirect_stderr(self.stderr):
            self.benchmark.run()
        self.assertEqual(run.call_count, 2)
        self.assertFalse(self.benchmark.using_installed)
        self.assertEqual(self.command.call_count, 4)
        self.assertEqual(self.command.call_args_list[0].args[0][:3],
                         ["git", "clone", "https://github.com/NVIDIA/nccl.git"])
        self.assertEqual(self.benchmark.env["NCCL_HOME"], str(self.root / "output" / "nccl-source" / "build"))
        self.assertEqual(self.benchmark.test_executable, binary)
        self.assertEqual(self.command.call_args_list[3].args[0], ["make", "-B"])
        self.assertEqual(self.command.call_args_list[3].kwargs["cwd"], self.root / "nccl-tests")
        self.assertIn("[y/N]", self.stderr.getvalue())

    def test_runtime_version_is_reported_on_cli_from_the_selected_library(self):
        include, libraries = self.installation()
        with redirect_stderr(self.stderr):
            runtime.configure_logging(self.root / "output")
        self.benchmark._configure_library(include, libraries)
        self.benchmark._report_version()
        command = self.command.call_args
        self.assertEqual(command.args[0][-1], str(libraries / "libnccl.so"))
        self.assertTrue(command.kwargs["text"])
        self.assertTrue(command.kwargs["capture_output"])
        self.assertIn("INFO Using NCCL 2.28.9", self.stderr.getvalue())
        self.assertIn("Using NCCL 2.28.9", (self.root / "output/run.log").read_text())

    def test_invalid_runtime_versions_are_explicit_errors(self):
        self.benchmark._configure_library(*self.installation())
        self.command.side_effect = None
        for version in ("", "unknown", "0", "-1"):
            with self.subTest(version=version):
                self.command.return_value = SimpleNamespace(stdout=version)
                with self.assertRaisesRegex(runtime.OutputError, "Invalid NCCL version"):
                    self.benchmark._report_version()

    def test_decline_eof_and_noninteractive_failure_do_not_rebuild(self):
        for response, interactive in (("no\n", True), ("", True), ("yes\n", False)):
            with self.subTest(response=response, interactive=interactive):
                self.benchmark.using_installed = True
                with patch.object(self.benchmark, "_run", side_effect=runtime.OutputError("invalid results")) as run, \
                        patch.object(self.benchmark, "_build_nccl") as source_build, \
                        patch.object(self.module.sys, "stdin", io.StringIO(response)) as stdin, \
                        patch.object(stdin, "isatty", return_value=interactive), redirect_stderr(self.stderr):
                    with self.assertRaises(runtime.CommandError):
                        self.benchmark.run()
                source_build.assert_not_called()
                run.assert_called_once()

    def test_build_failure_also_requests_approval(self):
        include, libraries = self.installation()
        with patch.object(self.module, "find_nccl", return_value=(include, libraries)), \
                patch.object(self.benchmark, "_build_tests", side_effect=[
                    runtime.CommandError("link failed"), None
                ]) as test_build, patch.object(self.benchmark, "_build_nccl") as source_build, \
                patch.object(self.module.sys, "stdin", io.StringIO("y\n")) as stdin, \
                patch.object(stdin, "isatty", return_value=True), redirect_stderr(self.stderr):
            self.benchmark.build()
        self.assertEqual(test_build.call_count, 2)
        source_build.assert_called_once()

    def test_rebuilt_failure_and_interrupt_are_not_retried(self):
        for error in (runtime.CommandError("rebuilt tests failed"), KeyboardInterrupt()):
            with self.subTest(error=error):
                self.benchmark.using_installed = False
                with patch.object(self.benchmark, "_run", side_effect=error) as run, \
                        patch.object(self.benchmark, "_build_nccl") as source_build:
                    with self.assertRaises(type(error)):
                        self.benchmark.run()
                run.assert_called_once()
                source_build.assert_not_called()

    def test_failed_retry_is_not_repeated(self):
        self.benchmark.using_installed = True
        with patch.object(self.benchmark, "_run", side_effect=runtime.CommandError("failed twice")) as run, \
                patch.object(self.benchmark, "_build_nccl"), patch.object(self.benchmark, "_build_tests"), \
                patch.object(self.module.sys, "stdin", io.StringIO("yes\n")) as stdin, \
                patch.object(stdin, "isatty", return_value=True), redirect_stderr(self.stderr):
            with self.assertRaisesRegex(runtime.CommandError, "failed twice"):
                self.benchmark.run()
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
