#!/bin/bash
# Installs Python environment dependencies for Azure AI Benchmarking.
set -euo pipefail

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [pip_command]

Arguments:
  pip_command     The pip executable to use (default: 'python3 -m pip')

Example:
  $0 'uv pip'          # Use 'uv pip' instead of 'python3 -m pip'
  $0                   # Defaults to 'python3 -m pip'
  $0 --help            # Show this help message
EOF
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

pip=${1:-'python3 -m pip'}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Determine GPU platform
if command -v rocminfo &> /dev/null && rocminfo &> /dev/null; then
    platform='AMD'
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    platform='NVIDIA'
else
    echo -e "\e[1;31m[Error] No NVIDIA or AMD GPU detected. Exiting.\e[0m" >&2
    exit 1
fi

echo "Using ${pip} to install AI Benchmarking Python dependencies for ${platform} platform"

# Warn if installing outside a virtual environment
if [ -z "${VIRTUAL_ENV:-}" ]; then
    printf "\033[1;33m[Warning] Installing dependencies outside of a virtual environment.\033[0m\n"
else
    echo "- Installing dependencies in virtual environment at: ${VIRTUAL_ENV}"
fi
echo ""

# Function to clone a repo if it doesn't exist
clone_repo() {
    local repo_url=$1
    local repo_dir=$2
    local checkout_commit=${3:-}

    if [ -d "$repo_dir" ]; then
        echo "$repo_dir already exists. Skipping clone."
    else
        git clone "$repo_url" "$repo_dir"
    fi

    pushd "$repo_dir" > /dev/null || { echo "Failed to enter $repo_dir"; exit 1; }
    [ -n "$checkout_commit" ] && git checkout "$checkout_commit"
    popd > /dev/null
}

# Install dependencies based on GPU platform
if [[ "$platform" == "AMD" ]]; then
    $pip install -r requirements/torch-amd.txt
    # Keep the ROCm-only index separate from common PyPI dependencies.
    $pip install -r requirements/common.txt -r requirements/build.txt

    echo "Building FlashAttention for AMD/ROCm"
    clone_repo "https://github.com/triton-lang/triton" "triton" "3ca2f498e98ed7249b82722587c511a5610e00c4"
    pushd triton > /dev/null
    $pip install -e python
    popd > /dev/null

    export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
    clone_repo "https://github.com/Dao-AILab/flash-attention.git" "flash-attention" "418d677"
    pushd flash-attention > /dev/null
    $pip install . --no-build-isolation --no-deps
    popd > /dev/null


elif [[ "$platform" == "NVIDIA" ]]; then
    gpu_output=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
    if echo "$gpu_output" | grep -q "GB"; then
        # Preserve the existing unpinned Blackwell Torch choice.
        $pip install -r requirements/common.txt -r requirements/torch-blackwell.txt
    else
        $pip install -r requirements/common.txt -r requirements/torch-nvidia.txt
        $pip install -r requirements/flashattn.txt --no-build-isolation --no-deps
    fi
fi

$pip check

# Warn if fio not installed
if ! command -v fio &> /dev/null; then
    printf "\033[1;33m[Warning] fio is not available in current PATH.\033[0m\n"
    printf "\033[1;33m[Warning] It may need to be installed to run the fio benchmark: sudo apt install fio\033[0m\n"
fi

# Huggingface home directory is PWD
export HF_HOME=$PWD
