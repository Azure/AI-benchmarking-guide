import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib import import_module
import json
import os
import re
from pathlib import Path
import sys
import time

from Infra import runtime, tools


ROOT = Path(__file__).resolve().parents[1]



def reserve_run_directory(parent, benchmark, timestamp):
    import fcntl

    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(benchmark)}_run_(\d+)_.*")
    with (parent / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        numbers = []
        for entry in parent.iterdir():
            match = pattern.fullmatch(entry.name)
            if match and entry.is_dir():
                numbers.append(int(match.group(1)))
        number = max(numbers, default=0) + 1
        while True:
            directory = parent / f"{benchmark}_run_{number}_{timestamp}"
            try:
                directory.mkdir()
            except FileExistsError:
                number += 1
            else:
                return directory


def run_benchmark(platform, name, machine):
    modules = {
        "NVIDIA": {
            "gemm": ("GEMMCublasLt", "GEMMCublastLt", ["build", "run_model_sizes"]),
            "nccl": ("NCCLBandwidth", "NCCLBandwidth", ["build", "run"]),
            "hbm": ("HBMBandwidth", "HBMBandwidth", ["build", "run"]),
            "nv": ("NVBandwidth", "NVBandwidth", ["build", "run"]),
            "fa": ("FlashAttention", "FlashAttention", ["run"]),
            "fio": ("FIO", "FIO", ["run"]),
            "cpustream": ("CPUStream", "CPUStream", ["build", "run"]),
            "multichase": ("Multichase", "Multichase", ["build", "run"]),
            "llm": ("LLMBenchmark", "LLMBenchmark",
                    ["install_requirements", "prepare_datasets", "download_models", "run_benchmark"]),
            "llama_8b_pretrain": ("LLAMA3Run", "LLAMA3Pretraining", ["run"]),
            "llama_3b_pretrain": ("LLAMA3Run", "LLAMA3Pretraining", ["run"]),
        },
        "AMD": {
            "gemm": ("GEMMHipblasLt", "GEMMHipBLAS", ["create_container", "build", "run_model_sizes"]),
            "rccl": ("RCCLBandwidth", "RCCLBandwidth", ["create_container", "build", "run"]),
            "hbm": ("HBMBandwidth", "HBMBandwidth", ["build", "run"]),
            "transfer": ("TransferBench", "TransferBench", ["build", "run"]),
            "fa": ("FlashAttention", "FlashAttention", ["run"]),
            "fio": ("FIO", "FIO", ["run"]),
            "llm": ("LLMBenchmark", "LLMBenchmark", ["create_container", "run_benchmark"]),
        },
    }
    if platform == "NVIDIA" and name == "hbm" and "GB200" in machine:
        raise runtime.UnsupportedHardwareError("HBM bandwidth is not yet supported on GB200")
    if name.startswith("llama_"):
        tools.get_pretraining_machine(machine)
    module_name, class_name, methods = modules[platform][name]
    try:
        module = import_module(f"Benchmarks.{platform}.{module_name}")
    except ImportError as exc:
        raise runtime.PrerequisiteError(
            f"Cannot load {name}: {exc}. Run install-dependencies.sh for this platform."
        ) from exc
    cls = getattr(module, class_name)
    config = str(ROOT / "config.json")
    if platform == "AMD" and name in ("fa", "hbm"):
        instance = cls(str(ROOT), machine)
    elif platform == "AMD":
        instance = cls(config, machine) if name == "fio" else cls("config.json", str(ROOT), machine)
    elif name == "llm":
        instance = cls(config, str(ROOT), machine)
    elif name.startswith("llama_"):
        instance = cls(config, machine, "3b" if name == "llama_3b_pretrain" else "8b")
    else:
        instance = cls(config, machine)
    with runtime.managed_container(instance) if platform == "AMD" else nullcontext():
        for method in methods:
            with runtime.stage(method.replace("_", " ")):
                getattr(instance, method)()


def detect_machine(platform):
    if platform == "NVIDIA":
        output = tools.run_command(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        if not output:
            raise runtime.PrerequisiteError("nvidia-smi returned no GPUs")
        return output.splitlines()[0].replace(" ", "_")
    output = tools.run_command(["rocminfo"], capture_output=True, text=True).stdout
    if "gfx942" not in output:
        raise runtime.UnsupportedHardwareError("AMD benchmarks currently require an MI300X (gfx942)")
    (tools.get_output_dir() / "system_specs.txt").write_text(output, encoding="utf-8")
    return "ND_MI300X_v5"



def format_run_summary(directory, rows, exit_code):
    overall = "INTERRUPTED" if exit_code == 130 else "FAILED" if exit_code else \
        "PASSED" if any(row["status"] == "PASS" for row in rows) else "SKIPPED"
    summary = ["Benchmark               Status        Duration"]
    for row in rows:
        duration = "-" if row["duration_s"] is None else f'{row["duration_s"]:.1f}s'
        summary.append(f'{row["benchmark"]:<23} {row["status"]:<13} {duration}')
        if row.get("output_dir"):
            summary.append(f'    {row["output_dir"]}')
    summary.extend([f"\nOverall: {overall}", f"Logs and results: {directory}"])
    return overall, "\n".join(summary)


def save_run_status(directory, platform, rows, exit_code):
    overall, _ = format_run_summary(directory, rows, exit_code)
    (directory / "status.json").write_text(json.dumps(
        {"platform": platform, "status": overall, "exit_code": exit_code, "benchmarks": rows},
        indent=2,
    ), encoding="utf-8")


def main(platform, argv=None, *, execute=None, discover=None):
    choices = {
        "NVIDIA": ["gemm", "nccl", "multichase", "cpustream", "hbm", "nv", "fio", "fa",
                   "llm", "llama_8b_pretrain", "llama_3b_pretrain"],
        "AMD": ["hbm", "transfer", "rccl", "fio", "fa", "llm", "gemm"],
    }[platform]
    parser = argparse.ArgumentParser(description=f"{platform} GPU benchmarks")
    parser.add_argument("benchmarks", nargs="+", type=str.lower, metavar="BENCHMARK",
                        help=f"One or more of: {', '.join(choices)}, all")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show commands and diagnostic output")
    verbosity.add_argument("--quiet", action="store_true", help="Show errors and the final summary only")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "Outputs",
                        help="Parent directory for numbered benchmark runs (default: Outputs)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Run remaining benchmarks after a failure; the overall exit status still fails")
    parser.add_argument("--timeout", type=float, metavar="SECONDS",
                        help="Timeout for each external command (default: no time limit)")
    args = parser.parse_args(argv)
    unknown = set(args.benchmarks) - set(choices) - {"all"}
    if unknown:
        parser.error(f"Unknown benchmarks: {', '.join(sorted(unknown))}")
    if args.timeout is not None and (not 0 < args.timeout < float("inf")):
        parser.error("--timeout must be a finite positive number")
    selected = choices if "all" in args.benchmarks else list(dict.fromkeys(args.benchmarks))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.output_dir.resolve()
    try:
        directory = reserve_run_directory(directory, selected[0] if len(selected) == 1 else "suite", timestamp)
        runtime.configure_logging(directory, verbose=args.verbose, quiet=args.quiet, timeout=args.timeout)
    except OSError as exc:
        runtime.close_logging()
        print(f"ERROR Cannot create output directory {directory}: {exc}", file=sys.stderr)
        return 1
    execute = execute or (lambda name, machine: run_benchmark(platform, name, machine))
    discover = discover or (lambda: detect_machine(platform))
    rows = []
    exit_code = 0
    original = Path.cwd()
    stopped = False
    try:
        with runtime.benchmark_context("setup"), runtime.stage("hardware discovery"):
            machine = discover()
        for index, name in enumerate(selected, 1):
            if stopped:
                rows.append({"benchmark": name, "status": "SKIPPED", "duration_s": None, "output_dir": None})
                continue
            started = time.monotonic()
            status = "PASS"
            benchmark_directory = directory if len(selected) == 1 else reserve_run_directory(
                args.output_dir, name, timestamp
            )
            with runtime.benchmark_context(name, benchmark_directory), tools.defer_markdown() as markdown:
                runtime.logger.info("[%s/%s] %s: starting", index, len(selected), name)
                try:
                    os.chdir(ROOT)
                    execute(name, machine)
                    if markdown:
                        with runtime.temporary_directory(prefix="results-", dir=benchmark_directory) as temporary:
                            pending = Path(temporary) / "summary.md"
                            pending.write_text("".join(markdown), encoding="utf-8")
                            pending.replace(benchmark_directory / "summary.md")
                except runtime.UnsupportedHardwareError as exc:
                    status = "SKIPPED"
                    runtime.logger.warning("%s", exc)
                except KeyboardInterrupt:
                    status, exit_code, stopped = "INTERRUPTED", 130, True
                    runtime.logger.error("%s: interrupted; full log: %s", name, runtime.log_path())
                except (runtime.BenchmarkError, OSError, ValueError, KeyError) as exc:
                    status, exit_code = "FAIL", 1
                    stopped = not args.continue_on_error
                    runtime.logger.error("%s failed during %s: %s; full log: %s", name, getattr(exc, "stage", "run"), exc, runtime.log_path())
                    runtime.logger.debug("Failure details", exc_info=True)
                except Exception:
                    status, exit_code = "FAIL", 1
                    stopped = not args.continue_on_error
                    runtime.logger.error("%s: internal error; full log: %s", name, runtime.log_path())
                    runtime.logger.debug("Unexpected failure", exc_info=True)
                finally:
                    os.chdir(original)
                elapsed = time.monotonic() - started
                rows.append({"benchmark": name, "status": status, "duration_s": round(elapsed, 3),
                             "output_dir": str(benchmark_directory)})
                if status == "PASS":
                    runtime.logger.info("%s: completed in %.1fs; results: %s", name, elapsed,
                                        benchmark_directory / "summary.md")
    except KeyboardInterrupt:
        exit_code = 130
        runtime.logger.error("Setup interrupted")
    except (runtime.BenchmarkError, OSError) as exc:
        exit_code = 1
        runtime.logger.error("Run setup or output allocation failed: %s", exc)
        runtime.logger.debug("Setup failure details", exc_info=True)
    except Exception:
        exit_code = 1
        runtime.logger.error("Internal error during setup; full log: %s", directory / "run.log")
        runtime.logger.debug("Unexpected setup failure", exc_info=True)
    finally:
        os.chdir(original)
        known = {row["benchmark"] for row in rows}
        rows.extend({"benchmark": name, "status": "SKIPPED", "duration_s": None, "output_dir": None}
                    for name in selected if name not in known)
        overall, text = format_run_summary(directory, rows, exit_code)
        try:
            for row in rows:
                if row.get("output_dir") and Path(row["output_dir"]) != directory:
                    code = 130 if row["status"] == "INTERRUPTED" else 1 if row["status"] == "FAIL" else 0
                    save_run_status(Path(row["output_dir"]), platform, [row], code)
            save_run_status(directory, platform, rows, exit_code)
        except OSError as exc:
            runtime.logger.error("Could not save run status: %s", exc)
            if exit_code == 0:
                exit_code = 1
                text = text.replace(f"Overall: {overall}", "Overall: FAILED")
        finally:
            print(text)
            runtime.logger.debug("%s", text)
            runtime.close_logging()
    return exit_code
