# Using Pyxis to build enroot-ified images

This directory contains all the shell scripts necessary to use pyxis to build enroot-ified container images on which one can run.

The makefile at `closed/NVIDIA/Makefile.pyxis` provides a user-friendly interface into these files.

There are two stages: 
1. a build stage, where a container will be built and serialized, and 
2. a run stage, where the container will be used to launch workloads

- [Stage 0: Preliminary](#stage-0-preliminary)
- [Stage 1: Building an image from a released container](#stage-1-building-an-image-from-a-released-container-such-as-in-ngc)
- [Stage 2: Running an image from a built container](#stage-2-running-an-image-from-a-built-container)
- [Relevant environment variables](#relevant-environment-variables)
- [Known Issues](#known-issues)
    - [Installing TRTLLM and Loadgen in run stage](#installing-trtllm-and-loadgen-in-run-stage)
    - [Running workloads](#running-workloads)
    - [TRTLLM build failure](#trtllm-build-failure-error-errno-17-file-exists-rootlocal)
    - [TRTLLM build fails with root user](#trtllm-build-failure-with-user-root)
- [TLDR](#tldr)

## Stage 0: Preliminary
It is required to export `MLPERF_SCRATCH_PATH` variable to point to a path where models, data and preprocessed_data reside. See `closed/NVIDIA/README.md` for more info.
```bash
$ export MLPERF_SCRATCH_PATH="/path/to/your/scratch/space"
$ export MLPERF_SCRATCH_PATH="/home/mlperf_inference_storage" # this is the default value
```

You may also export variables beforehand. For a list of variables, see [Relevant environment variables](#relevant-environment-variables).

## Stage 1: Building an image from a released container (such as in NGC):

We usually use `make prebuild` which depends on docker cli to:
- Pull a base image
- Build a final image from that base image, by installing dependancies

Now, the enroot equivalent command is:

```bash
$ make -f Makefile.pyxis build_base_sqsh ARCH=x86_64 SLURM_MODE="srun" EXTRA_SRUN_FLAGS="--container-remap-root" # runs a x86_64 container build job with `srun`
$ make -f Makefile.pyxis build_base_sqsh ARCH=aarch64 EXTRA_SRUN_FLAGS="--container-remap-root" # runs a aarch64 container build job with `srun`
$ make -f Makefile.pyxis build_base_sqsh ARCH=aarch64 SLURM_MODE="sbatch" EXTRA_SRUN_FLAGS="--container-remap-root" # runs a aarch64 container build job with `sbatch` to queue
```

This command should be run from a node which has `slurm` utility available to it - usually a login node. It then builds a required `sbatch` script or `srun` command which runs the build job on a compute node.

After the build job runs successfully, you should be able to find the enrootified image on the path you specified on `SBATCH_CONTAINER_SAVE`.
We can use this container to run workloads.

## Stage 2: Running an image from a built container:
In order to use run MLPerf Inference code on single or multi-node systems, it is necessary to build an enroot image with TensorRT LLM and MLPerf loadgen and save to file system (using steps above). Later on this image should be used by srun command with `--container-image` pyxis option.

In order to construct a consistent srun command, _it is strongly recommended_ to use `Makefile.pyxis`:
```bash
$ make -f Makefile.pyxis launch_sqsh_pty ARCH=aarch64 EXTRA_SRUN_FLAGS="--mpi=pmix"
```

The final srun command to be run looks something like: 

```bash
# An example srun command to use the serialized sqsh container
$ srun -p <Partition> -A <account> --mpi=pmix
    --container-image=<example enroot image: my_image.sqsh>
    --container-mounts=<path to MLPerf Inference>/mlperf-inference/closed/NVIDIA:/work --container-workdir=/work --pty bash
```

It is recommended (and essential for multi-node deployments) to use `sbatch` scripts instead of an srun session. 

## Relevant environment variables:
| Environment Variable | Values              | Build Stage `build_base_sqsh`   | Run Stage `launch_sqsh_pty`  | Notes |
|----------------------|---------------------|---------------------------------|------------------------------|-------|
| `ARCH`               | `x86_64`, `aarch64` | Required                        | Required                     | Must be defined |
| `ENV`                | `dev`, `release`    | Required    | Required | Defaults to declaration in `Makefile.const`. |
| `SLURM_MODE`         | `srun`, `sbatch` | Defaults to `srun`   | Defaults to `srun` (`sbatch` not supported) | `sbatch` may be buggy |
| `SBATCH_PARTITION` | Any valid partition name | Required | Required | Prompted if not defined |
| `SBATCH_CONTAINER_IMAGE` | File system path (e.g., `my_image.sqsh`) or docker registry tag| Required | Not required | Prompted if not defined |
| `SBATCH_CONTAINER_SAVE` | File system path (e.g., `my_image.sqsh`) | Not required | Not required | Prompted if not defined |
| `SBATCH_NODELIST` | Node names (e.g., `node[1-3]`) | Not required | Not required | Must be defined if needed |
| `INSTALL_TRTLLM` | `0`, `1` | Optional (Recommended: 1) | Ignored | |
| `INSTALL_LOADGEN` | `0`, `1` | Optional (Recommended: 1) | Ignored | |
| `MLPERF_SCRATCH_PATH` | Path to scratch space | Required | Required | Default is `/home/mlperf_inference_storage` |
| `EXTRA_SRUN_FLAGS` | Additional srun flags | Optional. Must specify `EXTRA_SRUN_FLAGS="--container-remap-root"` in build stage | Optional | Only works if `SLURM_MODE=srun` |

## Example: Building an initial image, and then modifying/adding installations.
Say we have built a base image as defined in [Stage 1](#stage-1-building-an-image-from-a-released-container-such-as-in-ngc). Now, we want to modify the environment in some way, for example, reinstall TRTLLM.

The way to do this is first launch a `Run stage` srun job: 
```bash
$ make -f Makefile.pyxis launch_sqsh_pty ARCH=aarch64 EXTRA_SRUN_FLAGS="--mpi=pmix --nodes=1"
```
You will be prompted for the existing CONTAINER_IMAGE: 
```bash
...
SBATCH_CONTAINER_IMAGE is not set.
Please provide a path to the sqsh file or docker image to pull: /path/to/old/image.sqsh
```
Type in the path to the existing `.sqsh` file.

Next, when prompted, give a path to the final modified image (`.sqsh` file) - you may give the same path to overwrite.
```bash
SBATCH_CONTAINER_SAVE is not defined. Please enter path to save sqsh file, leave blank to not save: /path/to/new/image.sqsh
```

Once the `srun` job launches and the terminal is active, modify the installations as necessary.

```bash
#  to reinstall TRTLLM
$ rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm
# to install anything else
$ python3 -m pip install sample_pip_package && apt-get install sample_apt_package
$ exit # end the bash session to serialize the sqsh file to /path/to/new/image.sqsh
```


## Known Issues
### Installing TRTLLM and Loadgen in run stage
Currently, with `INSTALL_TRTLLM=1 make -f build_base_sqsh`, we have noticed that the TRTLLM build may fail sometimes. If TRTLLM build fails during the build stage, the workaround is to install TRTLLM during the run stage:

```bash
$ make -f Makefile.pyxis build_base_sqsh ARCH=x86_64 INSTALL_TRTLLM=0 INSTALL_LOADGEN=0
# build and save a sqsh file to /path/to/container.sqsh
```

Start a run stage with the same sqsh file as a save target:
```bash
$ SBATCH_CONTAINER_IMAGE=/path/to/container.sqsh SBATCH_CONTAINER_SAVE=/path/to/container.sqsh ARCH=... make -f Makefile.pyxis launch_sqsh_pty
```

Then, from within the interactive slurm session: 
```bash
$ rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm
$ rm -rf build/inference && make clone_loadgen && make build_loadgen
$ exit # to invoke serialization of container to /path/to/container.sqsh
```

You can now use the `/path/to/container.sqsh` for running workloads.

### Running workloads
While you may rely on `make -f Makefile.pyxis launch_sqsh_pty` for a single node interactive session, it is recommended (and essential for multi-node deployments) to use `sbatch` scripts instead of an interactive session.  

We will share sbatch scripts that can reproduce workload runs as soon as possible in subsequent drops. For now, we recommend to copy the `srun` command built by the `launch_sqsh_pty` target and copy over the slurm flags to an equivalent sbatch script. An example is below

```bash
#!/bin/bash

#SBATCH --job-name=mlperf_inf_slurm_job
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --export=BASE_IMAGE=base_image_url_or_sqsh,ENV=dev,BUILD_CONTEXT=aarch64-Grace,CUDA_VER=12.9,TRT_VER=10.10.0.31,MITTEN_HASH=82a930d962ce6bd8aed38cf185a2acfbbfd6b84b,MITTEN_URL=,INSTALL_RGAT_DEPS=0,INSTALL_TRTLLM,INSTALL_LOADGEN,MLPERF_SCRATCH_PATH
#SBATCH --container-image=/path/to/image.sqsh/or/docker/tag/url
#SBATCH --container-mounts=....
#SBATCH --container-workdir=/work
#SBATCH --container-env=BASE_IMAGE,ENV,BUILD_CONTEXT,CUDA_VER,TRT_VER,MITTEN_HASH,MITTEN_URL,INSTALL_RGAT_DEPS,INSTALL_TRTLLM,INSTALL_LOADGEN,MLPERF_SCRATCH_PATH
#SBATCH --container-mount-home
#SBATCH --account=gtc_inference

```

We are working on providing workload specific single/multi-node sbatch scripts that can be used via `Makefile.pyxis`

### TRTLLM Build failure: `error: [Errno 17] File exists: '/root/.local'`
If you get the following error: 
```bash
error: [Errno 17] File exists: '/root/.local'
CMake Error at tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt:33 (message):
  Failed to set up the CUTLASS library due to 1.
```

The fix/workaround is:
```bash
rm /root/.local && rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm
```

### TRTLLM Build failure with user `root`
When we use the super-user `root` in the enroot container (using `--container-remap-root`) - the TRTLLM build _may_ fail due to failing cutlass compilation. In order to fix this, install TRTLLM in the run phase as defined in [Installing TRTLLM and LoadGen in run phase](#installing-trtllm-and-loadgen-in-run-stage) by using `--no-container-remap-root`. TLDR:

```bash
$ make -f Makefile.pyxis build_base_sqsh ARCH=x86_64 INSTALL_TRTLLM=0 INSTALL_LOADGEN=1 EXTRA_SRUN_FLAGS="--container-remap-root"
# save to /path/to/container.sqsh
$ make -f Makefile.pyxis launch_sqsh_pty ARCH=x86_64 EXTRA_SRUN_FLAGS="--no-container-remap-root"
# pull from /path/to/container.sqsh, save to /path/to/container_with_trtllm.sqsh

# And, from inside the container
$ rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm

# Finally, exit the job to trigger serialization
$ exit
```

## TLDR:

```bash
# First, try:
$ make -f Makefile.pyxis build_base_sqsh ARCH=x86_64 INSTALL_TRTLLM=1 INSTALL_LOADGEN=1 EXTRA_SRUN_FLAGS="--container-remap-root"

# If TRTLLM fails, then do:
$ make -f Makefile.pyxis build_base_sqsh ARCH=x86_64 INSTALL_TRTLLM=0 INSTALL_LOADGEN=1 EXTRA_SRUN_FLAGS="--container-remap-root"

# Followed by: 
$ make -f Makefile.pyxis launch_sqsh_pty ARCH=x86_64 EXTRA_SRUN_FLAGS="--no-container-remap-root"
# pull from /path/to/container.sqsh, save to /path/to/container_with_trtllm.sqsh

# And within the slurm job:
$ rm -rf build/TRTLLM && rm -f /root/.local && make clone_trt_llm && make build_trt_llm && exit
```