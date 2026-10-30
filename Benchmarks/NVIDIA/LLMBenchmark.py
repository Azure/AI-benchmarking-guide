import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from prettytable import PrettyTable

from Infra import tools


_TENSORRT_LLM_TAG = "v1.3.0rc14"


class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = tools.load_model_config(config_path, "nvidia")
        self.dir_path = Path(dir_path)
        self.machine = machine
        self.table = None
        for name in ("datasets", "engines", "hub"):
            (self.dir_path / name).mkdir(exist_ok=True)

    def models(self):
        return ((name, model) for name, model in self.config["models"].items()
                if model["use_model"] and model["type"] == "nvidia")

    def dataset_path(self, model_name, isl, osl):
        return self.dir_path / "datasets" / f"{model_name.split('/')[-1]}_synthetic_{isl}_{osl}.txt"

    def install_requirements(self):
        os.environ["HF_HOME"] = str(self.dir_path)
        repository = self.dir_path / "TensorRT-LLM"
        if not repository.exists():
            tools.run_command([
                "git", "clone", "--branch", _TENSORRT_LLM_TAG, "--single-branch",
                "https://github.com/NVIDIA/TensorRT-LLM.git", str(repository),
            ])
        else:
            tools.run_command(["git", "-C", str(repository), "fetch", "origin", "tag", _TENSORRT_LLM_TAG])
            tools.run_command(["git", "-C", str(repository), "checkout", "--detach", _TENSORRT_LLM_TAG])

    def download_models(self):
        for name, _ in self.models():
            tools.progress(f"Downloading {name}; gated models require Hugging Face login")
            tools.run_command(["huggingface-cli", "download", name, "--cache-dir", str(self.dir_path / "hub")])

    def prepare_datasets(self):
        for name, model in self.models():
            maximum = None
            maximum_size = 0
            for isl, osl in zip(model["input_sizes"], model["output_sizes"]):
                dataset = self.dataset_path(name, isl, osl)
                if isl + osl > maximum_size:
                    maximum, maximum_size = dataset, isl + osl
                if not dataset.exists() or dataset.stat().st_size == 0:
                    tools.progress(f"Preparing {name} dataset: {isl}/{osl}")
                    with NamedTemporaryFile(mode="w", dir=dataset.parent, delete=False) as file:
                        temporary = Path(file.name)
                        try:
                            tools.run_command([
                                "python3", str(self.dir_path / "TensorRT-LLM/benchmarks/cpp/prepare_dataset.py"),
                                "--stdout", "--tokenizer", name, "token-norm-dist",
                                "--num-requests", str(model["num_requests"]),
                                "--input-mean", str(isl), "--output-mean", str(osl),
                                "--input-stdev=0", "--output-stdev=0",
                            ], stdout=file, text=True)
                            if temporary.stat().st_size == 0:
                                raise tools.runtime.OutputError(f"Dataset generation returned no data for {name}")
                            temporary.replace(dataset)
                        finally:
                            temporary.unlink(missing_ok=True)
            if not (self.dir_path / "engines" / name).exists():
                tools.progress(f"Building engine for {name}")
                tools.run_command([
                    "trtllm-bench", "--workspace", str(self.dir_path / "engines"), "--model", name,
                    "build", "--tp_size", str(model["tp_size"]), "--dataset", str(maximum),
                    "--quantization", model["precision"],
                ])

    def run_benchmark(self):
        for name, model in self.models():
            self.table = PrettyTable(["tp size", "input len", "output len", "throughput(tokens/s)"])
            for isl, osl in zip(model["input_sizes"], model["output_sizes"]):
                tools.progress(f"Benchmarking {name}: input/output {isl}/{osl}, TP {model['tp_size']}")
                results = tools.get_output_dir() / f"results_{name.split('/')[-1]}_{isl}_{osl}.txt"
                with results.open("w") as file:
                    tools.run_command([
                        "trtllm-bench", "--model", name, "throughput",
                        "--backend", "tensorrt",
                        "--dataset", str(self.dataset_path(name, isl, osl)),
                        "--engine_dir", str(self.dir_path / "engines" / name / f"tp_{model['tp_size']}_pp_1"),
                        "--kv_cache_free_gpu_mem_fraction", "0.80",
                    ], stdout=file, text=True)
                self.extract_benchmark_info(results)
            tools.detail(self.table)
            tools.export_markdown(name, f"Precision: {model['precision']}; requests: {model['num_requests']}.", self.table)

    def extract_benchmark_info(self, file_path):
        keywords = [
            "TP Size:", "Average Input Length (tokens):",
            "Average Output Length (tokens):", "Token Throughput (tokens/sec):",
        ]
        values = {}
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                for keyword in keywords:
                    if keyword in line:
                        try:
                            value = float(line.split(keyword, 1)[1].strip())
                        except ValueError as exc:
                            raise tools.runtime.OutputError(f"Invalid {keyword} in {file_path}") from exc
                        if not math.isfinite(value) or value <= 0:
                            raise tools.runtime.OutputError(f"Invalid {keyword} {value} in {file_path}")
                        values[keyword] = str(value) if keyword == keywords[-1] else str(int(value))
        if set(values) != set(keywords):
            raise tools.runtime.OutputError(f"Missing LLM measurements in {file_path}: {set(keywords) - set(values)}")
        self.table.add_row([values[keyword] for keyword in keywords])
