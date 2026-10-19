# Common Issues FAQ
### What if I have permission issues when I attempt to write to the local directory from within the container?

This is most likely because your local directory is not writeable for `root`, or the user the docker daemon is launched with. This won't fix the root cause of the problem, which lies with the docker installation and how usergroups are set up, but you can workaround this by running `chmod -R 777` on the repo from outside the container.

To fix the root cause, try re-installing the latest Docker CE and following this [Docker guide from DigitalOcean](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket) on how to enable Docker for a non-root user. Namely, add your user to the Docker usergroup, and remove ~/.docker or chown it to your user. You may also have to restart the docker daemon for the changes to take effect:

```
$ sudo systemctl restart docker
```
### I get `useradd: user 'root' already exists` when running `make prebuild`.

As mentioned in the README.md, you should not run any commands involving our repo as root (this includes running make commands with sudo) unless explicitly specified.

### I get `Got permission denied while trying to connect to the Docker daemon socket` when attempting to launch the container.

Follow this [Docker guide from DigitalOcean](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket) on how to enable Docker for a non-root user. Namely, add your user to the Docker usergroup, and remove ~/.docker or chown it to your user. You may also have to restart the docker daemon for the changes to take effect:

```
$ sudo systemctl restart docker
```
### How do I access my local files from within the container?

Your user's home directory is mounted within the container at `/mnt/home/$USER`. The working directory (`/work`) is also a mounted volume of the working directory outside the container. You can also pass in additional flags to Docker by setting the `DOCKER_ARGS` environment variable:

```
$ make prebuild DOCKER_ARGS="-v my_dir:/my_dir"
```
### How do I install programs like valgrind inside the container?

The container's OS is Ubuntu 20.04, so you can simply use `apt`. If you want to make this permanent, you can add the package installs to the Dockerfile, located in at `closed/NVIDIA/docker/Dockerfile`.

### I get `nvcc fatal: Unsupported gpu architecture 'compute_XX'` when attempting to run `make build`.

Try running a clean build (`make clean && make build`). This error occurs when the prior call of `make build` had a different active CUDA version when compiling.

### Models and data are disappearing or being written to the wrong places. Scratch path isn't being linked correctly.

Make sure that the directory you set as `MLPERF_SCRATCH_PATH` has write permissions for user group `other`. `777` is the most generic option. This error has also been seen when executing make commands in the container using via other users (i.e. using `sudo`), as this does not inherit the `MLPERF_SCRATCH_PATH` environment variable from the current user.

### What are the power metrics relevant to the benchmark scenarios?

- For offline: samples/sec/watt or queries/sec/watt
- For server: samples/sec/watt or queries/sec/watt
- For single stream: joules/query
- For multi stream: joules/query

