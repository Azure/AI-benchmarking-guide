from contextlib import contextmanager
from contextvars import ContextVar
import os
import socket
import math
from Infra import runtime

AMD_BENCHMARK_IMAGE = "rocm/vllm-dev:main"

run_command = runtime.run_command
run_container_command = runtime.run_container_command
load_config = runtime.load_config
stage = runtime.stage
start_container = runtime.start_container
get_output_dir = runtime.output_dir
logger = runtime.logger
_markdown_buffer = ContextVar("markdown_buffer", default=None)


# Collect Markdown exports in a scoped buffer instead of writing them.
@contextmanager
def defer_markdown():
    contents = []
    token = _markdown_buffer.set(contents)
    try:
        yield contents
    finally:
        _markdown_buffer.reset(token)

# Validate all five BabelStream operations and return them in display order.
def parse_stream_measurements(output):
    operations = ("Copy", "Mul", "Add", "Triad", "Dot")
    measurements = {}
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] in operations:
            try:
                value = float(fields[1])
            except (ValueError, IndexError) as exc:
                raise runtime.OutputError(f"Invalid BabelStream measurement: {line}") from exc
            if not math.isfinite(value) or value <= 0:
                raise runtime.OutputError(f"Invalid BabelStream bandwidth: {line}")
            measurements[fields[0]] = [fields[0], str(value)]
    if set(measurements) != set(operations):
        raise runtime.OutputError(f"Missing BabelStream measurements: {set(operations) - set(measurements)}")
    return [measurements[name] for name in operations]

# Mount the benchmark directory and active output directory read-write.
def container_volumes(directory):
    return {
        str(path): {"bind": str(path), "mode": "rw"}
        for path in (directory, get_output_dir())
    }

# Build fresh shared AMD launch options with optional per-benchmark settings.
def amd_container_options(directory, *, entrypoint=None, environment=None):
    options = {
        "ipc_mode": "host",
        "network": "host",
        "group_add": ["render"],
        "privileged": True,
        "security_opt": ["seccomp=unconfined"],
        "cap_add": ["CAP_SYS_ADMIN", "SYS_PTRACE"],
        "devices": ["/dev/kfd", "/dev/dri", "/dev/mem"],
        "volumes": container_volumes(directory),
        "tty": True,
        "detach": True,
        "auto_remove": True,
    }
    if entrypoint is not None:
        options["entrypoint"] = entrypoint
    if environment is not None:
        options["environment"] = environment
    return options

# Load LLM configuration and validate enabled models for the given platform.
def load_model_config(path, platform):
    config = load_config(path, "LLMBenchmark")
    models = config.get("models")
    if not isinstance(models, dict):
        raise runtime.ConfigurationError("LLMBenchmark.models must be an object")
    selected = []
    for name, model in models.items():
        if not isinstance(model, dict):
            raise runtime.ConfigurationError(f"Invalid model configuration: {name}")
        if type(model.get("use_model")) is not bool or model.get("type") not in ("amd", "nvidia"):
            raise runtime.ConfigurationError(f"{name}: use_model must be a boolean and type must be amd or nvidia")
        if model.get("type") != platform or not model.get("use_model"):
            continue
        selected.append(name)
        input_key, output_key = ("input_sizes", "output_sizes") if platform == "nvidia" else ("input_length", "output_length")
        inputs, outputs = model.get(input_key), model.get(output_key)
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
            raise runtime.ConfigurationError(f"{name}: input/output lengths must be nonempty lists of equal length")
        fields = {input_key: inputs, output_key: outputs}
        for key in ("tp_size", "num_requests") if platform == "nvidia" else ("tp_sizes", "num_requests", "max_num_seqs"):
            value = model.get(key)
            fields[key] = [value] if platform == "nvidia" else value
        for key, values in fields.items():
            if not isinstance(values, list) or not values or any(type(value) is not int or value < 1 for value in values):
                raise runtime.ConfigurationError(f"{name}: {key} must contain positive integers")
        if platform == "nvidia" and not isinstance(model.get("precision"), str):
            raise runtime.ConfigurationError(f"{name}: precision must be a string")
    if not selected:
        raise runtime.ConfigurationError(f"No {platform} LLM models are enabled")
    return config

# Ensure a directory exists relative to the working directory and return its path.
def create_dir(name: str):
    current = os.getcwd()
    outdir = os.path.join(str(current), name)
    isdir = os.path.isdir(outdir)
    if not isdir:
        os.mkdir(outdir)
    return outdir

# Write a diagnostic message to the shared debug logger.
def write_log(message: str):
    logger.debug("%s", message)

# Join message parts into a user-visible progress log entry.
def progress(*parts):
    logger.info(" ".join(str(part) for part in parts))

# Join message parts into a debug-only diagnostic entry.
def detail(*parts):
    logger.debug(" ".join(str(part) for part in parts))

# Decode stderr when present, otherwise stdout; this does not check exit status.
def check_error(results):
    if results.stderr:
        return results.stderr.decode("utf-8")
    return results.stdout.decode("utf-8")

# Normalize supported GPU names for pretraining or reject unsupported hardware.
def get_pretraining_machine(machine_name):
    for supported in ("GB200", "H200"):
        if supported in machine_name:
            return supported
    raise runtime.UnsupportedHardwareError(f"LLAMA3 pretraining supports H200 and GB200; detected {machine_name}")

# Return the local machine's hostname.
def get_hostname():
    return socket.gethostname()

# Render a PrettyTable as Markdown, or return empty text when no table is given.
def prettytable_to_markdown(table):
    if table is None:
        return ""
    header = "| " + " | ".join(table.field_names) + " |"
    separator = "| " + " | ".join("---" for _ in table.field_names) + " |"
    rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table.rows]
    return "\n".join([header, separator] + rows) 

# Write or buffer Markdown results, then log and print the rendered output.
def export_markdown(title, description, table = None):
    content = ("## " + title + "\n\n" if title is not None else "")
    content += description + "\n" + prettytable_to_markdown(table) + "\n\n"
    pending = _markdown_buffer.get()
    if pending is None:
        filename = get_output_dir() / "summary.md"
        with open(filename, "a", encoding="utf-8") as file:
            file.write(content)
    else:
        pending.append(content)
    rendered_table = str(table) if table is not None else None
    results = "\n".join(part for part in (
        title, description, rendered_table
    ) if part is not None)
    logger.debug("Results:\n%s", results)
    print(runtime.redact(rendered_table if rendered_table is not None else results),
          end="\n\n", flush=True)
