#!/bin/bash
set -euo pipefail

m=$1
n=$2
k=$3
iterations=$4
warmup_iterations=$5
output_dir=${BENCHMARK_OUTPUT_DIR:-../../Outputs}

../../hipBLASLt/build/release/clients/staging/hipblaslt-bench --device 0 --flush --yaml - <<< "- {function: matmul, transA: T, transB: N, a_type: f8_r, b_type: f8_r, c_type: f16_r, d_type: f16_r, compute_type: c_f32_r, M: $m, N: $n, K: $k, lda: $k, ldb: $k, ldc: $m, ldd: $m, alpha: 1, beta: 0, scale_type: f32_r, iters: $iterations, cold_iters: $warmup_iterations, initialization: trig_float,rotating: 512}" | grep -B 1 "T,N,0" >> "$output_dir/GEMMHipBLAS_results.txt"
