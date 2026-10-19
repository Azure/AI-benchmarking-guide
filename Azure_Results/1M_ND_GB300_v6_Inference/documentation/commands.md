# MAKE Targets and RUN_ARGS Documentation
Our Makefile includes many targets and options. Below are some of the more relevant and commonly used ones:

### Make Targets

To execute outside the shell:

- `make prebuild`: Runs the following steps:
    - `make build_docker`: Builds docker image.
    - `make docker_add_user`: Adds current user to the docker image.
    - `make attach_docker`: Runs docker image with an interactive session and with current working directory bound to `/work` in the container. You are left with an interactive shell inside the container at `/work`.
- `make stage_results`: Updates the results/ directory with the logs in `build/logs`. Run during the submission process.
- `make check_submission`: Runs the official submission checker on the current state of the repo. Run during the submission process.

These `make` commands are run inside the Docker container shell, that you get by running `make prebuild` (see above):

- `make download_dataset`: Downloads datasets.
- `make preprocessed_data`: Preprocesses the downloaded datasets.
- `make build`: Runs the following steps:
    - `make link_dirs`: Adds symlinks in `build/` to the relevant directories from `$MLPERF_SCRATCH_PATH`
    - `make clone_loadgen`: Clone the official MLPerf inference GitHub repo.
    - `make build_plugins`: Builds TensorRT plugins.
    - `make build_loadgen`: Builds LoadGen source codes.
    - `make build_harness`: Builds the harnesses.
- `make run RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> [OTHER RUN_ARGS FLAGS]"` runs the following steps:
    - `make generate_engines RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> [OTHER RUN_ARGS FLAGS]"`: Generates TensorRT optimized engine files.
    - `make run_harness RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> [OTHER RUN_ARGS FLAGS]"`: Runs harnesses with engine files and LoadGen.
    - See RUN_ARGS flags.
- `make calibrate RUN_ARGS="--benchmarks=<BENCHMARKS>"`: Generates calibration caches.
    - See RUN_ARGS flags.
- `make clean`: Cleans up all the build directories. Please note that you will need to exit the docker container and run `make prebuild` again after the cleaning.
- `make clean_shallow`: Cleans up only the files needed to make a clean build. This includes any binaries and files that require compilation.

### RUN_ARGS Flags

- `--benchmarks=comma,separated,list,of,benchmark,names`
- `--scenarios=comma,separated,list,of,scenario,names`
- `--config_ver=comma,separated,list,of,config,versions`
- `--test_mode=[PerformanceOnly,AccuracyOnly]`: Specifies which LoadGen mode to run with.
- `--no_gpu`: Disables generating / running GPU engine (Tegra only).
- `--gpu_only`: Disables generating / running DLA engine (Tegra only).
- `--force_calibration`: Forces recalculation of calibration cache.
- `--log_dir=path/to/logs`: Specifies where to save logs.
- `--log_copy_detail_to_stdout`: Prints LoadGen detailed logs to stdout as well as to the log files.
- `--verbose`: Prints out verbose logs.

### Config Versions

There are four types of config versions which can be passed into the `--config_ver` flag:

1. `default`: This is the default config, as shown in `config.json`. No overrides are applied Uses custom harness with low accuracy target. Supported for all benchmarks.
2. `high_accuracy`: Runs the benchmark for the 99.9% of FP32 accuracy target. Uses custom harness with high accuracy target. Supported only for BERT, DLRM, and 3D-Unet.
3. `triton`: Uses Triton Inference Server with low accuracy target instead of the LWIS / Ninja / Custom harness. Supported for all benchmarks, except BERT. Only supported in server mode.
4. `high_accuracy_triton`: Uses Triton Inference Server to run the benchmark for the 99.9% of FP32 accuracy target. Supported only for DLRM and 3D-Unet.

`all`: Runs every `config_ver` available in `config.json`. This will override all other `config_ver` options specified by the flag.

Passing in comma-separated list of config versions or passing in `all` allows you to generate the engines or to run harness for multiple config versions in one command.

