import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
STUB = r"""import json
import os
from pathlib import Path
import sys
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["INSTALL_LOG"], "a") as file:
    file.write(json.dumps({"name": name, "args": args, "cwd": str(Path.cwd())}) + "\n")
if name == "rocminfo":
    sys.exit(0 if os.environ["TEST_GPU"] == "AMD" else 1)
if name == "nvidia-smi":
    print(os.environ["TEST_GPU"])
if name == "git" and args[0] == "clone":
    Path(args[-1]).mkdir()
if name == "record-pip":
    for index, argument in enumerate(args):
        if argument == "-r" and not Path(args[index + 1]).is_file():
            sys.exit(9)
    if args == ["check"] and os.environ.get("FAIL_CHECK"):
        sys.exit(7)
"""


class DependencyTests(unittest.TestCase):
    def run_installer(self, gpu, fail_check=False):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project with spaces"
            project.mkdir()
            shutil.copy2(ROOT / "install-dependencies.sh", project)
            shutil.copytree(ROOT / "requirements", project / "requirements")
            binaries = root / "bin"
            binaries.mkdir()
            for name in ("rocminfo", "nvidia-smi", "git", "fio", "record-pip"):
                executable = binaries / name
                executable.write_text(f"#!{sys.executable}\n" + STUB)
                executable.chmod(0o700)
            log = root / "commands.jsonl"
            env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.defpath,
                   "TEST_GPU": gpu, "INSTALL_LOG": str(log), "VIRTUAL_ENV": str(root / "venv")}
            if fail_check:
                env["FAIL_CHECK"] = "1"
            result = subprocess.run(
                ["bash", str(project / "install-dependencies.sh"), str(binaries / "record-pip")],
                cwd=root, env=env, capture_output=True, text=True, timeout=10,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, calls

    def test_nvidia_installs_torch_and_build_tools_before_flash_attention(self):
        result, calls = self.run_installer("NVIDIA H200")
        self.assertEqual(result.returncode, 0, result.stderr)
        pip = [call["args"] for call in calls if call["name"] == "record-pip"]
        self.assertEqual(pip[0], ["install", "-r", "requirements/common.txt", "-r", "requirements/torch-nvidia.txt"])
        self.assertEqual(pip[1], ["install", "-r", "requirements/flashattn.txt", "--no-build-isolation", "--no-deps"])
        self.assertEqual(pip[2], ["check"])

    def test_blackwell_reuses_common_pins_without_forcing_nvidia_torch_pin(self):
        for gpu in ("NVIDIA GB200", "NVIDIA GB300"):
            with self.subTest(gpu=gpu):
                result, calls = self.run_installer(gpu)
                self.assertEqual(result.returncode, 0, result.stderr)
                pip = [call["args"] for call in calls if call["name"] == "record-pip"]
                self.assertEqual(pip, [["install", "-r", "requirements/common.txt", "-r", "requirements/torch-blackwell.txt"], ["check"]])

    def test_amd_installs_common_dependencies_before_matching_source_revision(self):
        result, calls = self.run_installer("AMD")
        self.assertEqual(result.returncode, 0, result.stderr)
        pip = [call for call in calls if call["name"] == "record-pip"]
        self.assertEqual(pip[0]["args"], ["install", "-r", "requirements/torch-amd.txt"])
        self.assertEqual(pip[1]["args"], ["install", "-r", "requirements/common.txt", "-r", "requirements/build.txt"])
        self.assertEqual(pip[3]["args"], ["install", ".", "--no-build-isolation", "--no-deps"])
        self.assertTrue(pip[3]["cwd"].endswith("flash-attention"))
        self.assertIn(["checkout", "418d677"], [call["args"] for call in calls if call["name"] == "git"])
        self.assertEqual(pip[-1]["args"], ["check"])

    def test_dependency_check_failure_is_propagated(self):
        result, _ = self.run_installer("NVIDIA H200", fail_check=True)
        self.assertEqual(result.returncode, 7)

    def test_profiles_have_no_conflicting_exact_pins(self):
        def read_requirements(path):
            entries = []
            for line in path.read_text().splitlines():
                if line.startswith("-r "):
                    entries.extend(read_requirements(path.parent / line[3:]))
                elif line and not line.startswith(("#", "--")):
                    entries.append(line)
            return entries
        for profile in ("nvidia", "amd", "blackwell"):
            pins = {}
            entries = read_requirements(ROOT / "requirements/common.txt")
            entries += read_requirements(ROOT / f"requirements/torch-{profile}.txt")
            for entry in entries:
                match = re.fullmatch(r"([\w-]+)==([^=]+)", entry)
                if match:
                    name, version = match.groups()
                    self.assertEqual(pins.get(name, version), version, name)
                    pins[name] = version
            self.assertNotIn("sympy", pins)
            self.assertEqual(pins["packaging"], "24.2")

    def test_requirement_profiles_are_grouped_in_one_directory(self):
        self.assertEqual(sorted(path.name for path in (ROOT / "requirements").glob("*.txt")), [
            "build.txt", "common.txt", "flashattn.txt",
            "torch-amd.txt", "torch-blackwell.txt", "torch-nvidia.txt",
        ])
        self.assertEqual(list(ROOT.glob("requirements_*.txt")), [])
