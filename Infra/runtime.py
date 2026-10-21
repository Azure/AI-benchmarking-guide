import codecs
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from uuid import uuid4


class BenchmarkError(Exception):
    pass


class ConfigurationError(BenchmarkError, ValueError):
    pass


class PrerequisiteError(BenchmarkError):
    pass


class UnsupportedHardwareError(BenchmarkError, ValueError):
    pass


class OutputError(BenchmarkError, ValueError):
    pass


class CommandError(BenchmarkError, RuntimeError):
    pass


class CommandTimeoutError(CommandError):
    pass


logger = logging.getLogger("benchmarks")
logger.addHandler(logging.NullHandler())
_benchmark = ContextVar("benchmark", default="setup")
_stage = ContextVar("stage", default="setup")
_output_override = ContextVar("output_directory", default=None)
_output_dir = None
_timeout = None


def redact(message, env=None):
    message = str(message)
    for key, value in {**os.environ, **(env or {})}.items():
        if value and re.search(r"(TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_KEY)$", key, re.I):
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"\bhf_[A-Za-z0-9]+", "[REDACTED]", message)
    message = re.sub(r"(authorization[:=]\s*)(?:Bearer\s+)?\S+", r"\1[REDACTED]", message, flags=re.I)
    return re.sub(
        r"((?:--(?:token|password|api-key)|authorization[:=]|(?:HF_TOKEN|API_KEY|PASSWORD)=)\s*)\S+",
        r"\1[REDACTED]", message, flags=re.I,
    )


class SafeFormatter(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.benchmark = _benchmark.get()
        record.stage = _stage.get()
        return True


class BenchmarkFilter(logging.Filter):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def filter(self, record):
        return _benchmark.get() == self.name


def output_dir():
    path = _output_override.get() or _output_dir or Path.cwd() / "Outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path():
    return output_dir() / f"{_benchmark.get()}.log"


def configure_logging(path, *, verbose=False, quiet=False, timeout=None):
    global _output_dir, _timeout
    close_logging()
    _output_dir = Path(path).resolve()
    _timeout = timeout
    output_dir()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO)
    console.setFormatter(SafeFormatter("%(levelname)s %(message)s"))
    file_handler = logging.FileHandler(output_dir() / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(SafeFormatter(
        "%(asctime)s %(levelname)s benchmark=%(benchmark)s stage=%(stage)s %(message)s"
    ))
    for handler in (console, file_handler):
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)


def close_logging():
    global _output_dir, _timeout
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _output_dir = None
    _timeout = None


@contextmanager
def benchmark_context(name, directory=None):
    output_directory = Path(directory).resolve() if directory is not None else _output_override.get()
    token = _benchmark.set(name)
    output_token = _output_override.set(output_directory)
    handler = None
    try:
        handler = logging.FileHandler(output_dir() / f"{name}.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(SafeFormatter(
            "%(asctime)s %(levelname)s stage=%(stage)s %(message)s"
        ))
        handler.addFilter(BenchmarkFilter(name))
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)
        yield
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
        _output_override.reset(output_token)
        _benchmark.reset(token)


@contextmanager
def stage(name):
    token = _stage.set(name)
    message = "checking build requirements" if name == "build" else name
    logger.info("%s: %s...", _benchmark.get(), message)
    try:
        yield
    except BaseException as exc:
        logger.debug("Stage %s failed", name, exc_info=True)
        if isinstance(exc, BenchmarkError):
            exc.stage = name
        raise
    finally:
        _stage.reset(token)


def _enqueue(events, item, stopped):
    while not stopped.is_set():
        try:
            events.put(item, timeout=0.05)
            return
        except queue.Full:
            continue


def _reader(stream, label, events, stopped):
    try:
        while not stopped.is_set():
            chunk = stream.read1(4096)
            if not chunk:
                break
            _enqueue(events, (label, chunk), stopped)
    except OSError as exc:
        _enqueue(events, ("reader_error", exc), stopped)
    finally:
        _enqueue(events, (label, None), stopped)


def run_command(command, *, cwd=None, env=None, timeout=None, shell=False,
                capture_output=False, text=False, stdout=None, stderr=None, check=True):
    if not check:
        raise ValueError("Unchecked benchmark commands are not supported")
    if shell:
        command = ["bash", "-o", "pipefail", "-c", command]
    elif isinstance(command, str):
        command = shlex.split(command)
    displayed = redact(shlex.join(str(part) for part in command), env)
    logger.debug("command=%s cwd=%s", displayed, cwd or Path.cwd())
    started = time.monotonic()
    limit = _timeout if timeout is None else timeout
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise PrerequisiteError(
            f"Executable not found: {command[0]}. Install it or check PATH."
        ) from exc
    events = queue.Queue(maxsize=64)
    stopped = threading.Event()
    threads = [
        threading.Thread(target=_reader, args=(process.stdout, "stdout", events, stopped), daemon=True),
        threading.Thread(target=_reader, args=(process.stderr, "stderr", events, stopped), daemon=True),
    ]
    for thread in threads:
        thread.start()
    decoders = {name: codecs.getincrementaldecoder("utf-8")(errors="replace")
                for name in ("stdout", "stderr")}
    tail = ""
    finished = set()
    with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as out, \
            tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as err:
        buffers = {"stdout": out, "stderr": err}
        try:
            while len(finished) < 2 or process.poll() is None:
                if limit is not None and time.monotonic() - started > limit:
                    raise CommandTimeoutError(
                        f"Command exceeded {limit:g}s: {displayed}. Full log: {log_path()}"
                    )
                try:
                    label, chunk = events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if label == "reader_error":
                    raise CommandError(f"Could not read command output: {chunk}") from chunk
                if chunk is None:
                    finished.add(label)
                    continue
                if capture_output or (label == "stdout" and stdout == subprocess.PIPE) or \
                        (label == "stderr" and stderr == subprocess.PIPE):
                    buffers[label].write(chunk)
                decoded = decoders[label].decode(chunk)
                logger.debug("%s: %s", label, redact(decoded.rstrip(), env))
                tail = (tail + decoded)[-4000:]
                destination = stdout if label == "stdout" or stderr == subprocess.STDOUT else stderr
                if hasattr(destination, "write"):
                    destination.write(decoded if text else chunk)
                    destination.flush()
            code = process.wait()
        finally:
            stopped.set()
            if process.poll() is None or len(finished) < 2:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    logger.debug("Command process group already exited")
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.debug("Escalating command shutdown after 5 seconds")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    logger.debug("Command process group shutdown complete")
                process.wait()
            for thread in threads:
                thread.join()
            process.stdout.close()
            process.stderr.close()
        logger.debug("exit_code=%s elapsed_s=%.3f command=%s",
                     code, time.monotonic() - started, displayed)
        if code:
            raise CommandError(
                f"Command exited with code {code}: {displayed}\n"
                f"{redact(tail.strip(), env)}\nFull log: {log_path()}"
            )
        out.seek(0)
        err.seek(0)
        output, errors = out.read(), err.read()
    if text:
        output, errors = output.decode("utf-8", errors="replace"), errors.decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(command, code, output, errors)


def run_container_command(container, command, *, timeout=None):
    if isinstance(command, str):
        command = shlex.split(command)
    return run_command(
        ["docker", "exec", "--env", f"BENCHMARK_OUTPUT_DIR={output_dir()}", container.id, *command],
        capture_output=True, text=True, timeout=timeout,
    ).stdout


def start_container(image, **options):
    try:
        import docker
        from docker.errors import DockerException, ImageNotFound
    except ImportError as exc:
        raise PrerequisiteError("Docker SDK is missing; run install-dependencies.sh") from exc

    name = "ai-benchmark-" + uuid4().hex
    options["name"] = name
    options["auto_remove"] = True
    try:
        client = docker.from_env(timeout=_timeout or 60)
        try:
            client.images.get(image)
        except ImageNotFound:
            logger.info("Docker image %s is not available locally; pulling...", image)
            run_command(["docker", "pull", image])
        else:
            logger.info("Using local Docker image %s", image)
        with docker_run_cleanup(name):
            return client.containers.run(image, **options)
    except DockerException as exc:
        raise CommandError(f"Could not start container {image}: {exc}") from exc


@contextmanager
def managed_container(benchmark):
    try:
        yield
    except BaseException:
        try:
            cleanup_container(benchmark)
        except Exception as exc:
            logger.error("Container cleanup also failed; preserving the original failure: %s", exc)
            logger.debug("Container cleanup details", exc_info=True)
        raise
    else:
        cleanup_container(benchmark)


def cleanup_container(benchmark):
    container = getattr(benchmark, "container", None)
    if container is not None:
        benchmark.container = None
        logger.debug("Stopping container %s", container.id)
        container.kill()


@contextmanager
def docker_run_cleanup(name):
    try:
        yield
    except BaseException:
        try:
            result = run_command(
                ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
                capture_output=True, text=True, timeout=30,
            )
            if result.stdout.strip():
                run_command(["docker", "rm", "--force", name], timeout=30)
        except Exception as exc:
            logger.error("Docker cleanup also failed; preserving the original failure: %s", exc)
            logger.debug("Docker cleanup details", exc_info=True)
        raise



@contextmanager
def temporary_directory(**options):
    directory = tempfile.TemporaryDirectory(**options)
    try:
        yield directory.name
    except BaseException:
        try:
            directory.cleanup()
        except OSError as exc:
            logger.error("Temporary-file cleanup also failed; preserving the original failure: %s", exc)
            logger.debug("Temporary-file cleanup details", exc_info=True)
        raise
    else:
        directory.cleanup()


def load_config(path, section):
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        config = data[section]
        if not isinstance(config, dict):
            raise TypeError("section must be an object")
        return config
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigurationError(f"Invalid configuration {path}, section {section}: {exc}") from exc
