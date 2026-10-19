import re
import math
from prettytable import PrettyTable
from Infra import tools

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = self.get_config(config_path)
        self.dir_path = dir_path
        self.table = None
        self.container = None
        self.machine = machine

    def get_config(self, path: str):
        return tools.load_model_config(path, "amd")

    def create_container(self):
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'entrypoint':'/bin/bash',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': tools.container_volumes(self.dir_path),
            'environment': {'HF_HOME': str(self.dir_path)},
            'tty': True,
            'detach': True,
            'auto_remove': True
        }

        # Creates new Docker container
        tools.progress("Pulling docker container rocm/vllm-dev:20241121-tuned")
        self.container = tools.start_container('rocm/vllm-dev:20241121-tuned', **docker_run_options)
        tools.progress(f"Docker Container ID: {self.container.id}")

    def run_benchmark(self):
        with tools.managed_container(self):
            self._run_models()

    def _run_models(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "amd":
                self.table = PrettyTable(["input len", "output len", "tp size", "throughput(tokens/s)"])
                for tp_size in self.config['models'][model_name]['tp_sizes']:
                    tools.progress(f"Benchmarking {model_name} with TP Size: {tp_size}")
                    for max_num_seq in self.config['models'][model_name]['max_num_seqs']:
                        for i in range(len(self.config['models'][model_name]['input_length'])):
                            for request in self.config['models'][model_name]['num_requests']:
                                input_size = self.config['models'][model_name]['input_length'][i]
                                output_size = self.config['models'][model_name]['output_length'][i]
                                tools.progress(f" Input Size: {input_size}, Output Size: {output_size}...")
                                run_benchmark_command = f'''
                                    /bin/bash -c \
                                    "python /app/vllm/benchmarks/benchmark_throughput.py \
                                        --model amd/{model_name} \
                                        --quantization fp8 \
                                        --kv-cache-dtype fp8 \
                                        --dtype half \
                                        --gpu-memory-utilization 0.90 \
                                        --distributed-executor-backend mp \
                                        --num-scheduler-steps 10 \
                                        --tensor-parallel-size {tp_size} \
                                        --enable-chunked-prefill false \
                                        --max-seq-len-to-capture 131072 \
                                        --max-num-batched-tokens 131072 \
                                        --max-model-len 8192 \
                                        --max-num-seqs {max_num_seq} \
                                        --num-prompts {request} \
                                        --input-len {input_size} \
                                        --output-len {output_size}"
                                    '''
                                output = tools.run_container_command(self.container, run_benchmark_command)
                                match = re.search(r"(\S+)\s+output tokens/s", output)
                                if not match:
                                    raise tools.runtime.OutputError(f"No throughput measurement returned for {model_name}")
                                result = match.group(1)
                                try:
                                    throughput = float(result)
                                except ValueError as exc:
                                    raise tools.runtime.OutputError(f"Invalid throughput for {model_name}: {result}") from exc
                                if not math.isfinite(throughput) or throughput <= 0:
                                    raise tools.runtime.OutputError(f"Invalid throughput for {model_name}: {result}")
                                self.table.add_row([str(input_size), str(output_size), str(tp_size), result])
                tools.detail(self.table)
                tools.export_markdown(model_name, "Performance results with FP8 quantization.", self.table)
