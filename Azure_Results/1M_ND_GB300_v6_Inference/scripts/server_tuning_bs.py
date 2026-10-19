# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import subprocess as sp
import sys

# Needs to be changed for each system
# upper bound is offline qps
RESNET50_UPPER = 78000
BERT_LA_UPPER = 7000
BERT_HA_UPPER = 6000
DLRM_UPPER = 145000000 / 270

# Needs to be changed for each system
# User needs to indicate which batch sizes to check
benchmarks = [
    ('resnet50', 'default', RESNET50_UPPER, (128, 64, 256, 32, 512), {}),
    ('bert', 'default', BERT_LA_UPPER, (48, 32, 64, 16), {}),
    ('bert', 'high_accuracy', BERT_HA_UPPER, (64, 48, 128, 32, 256), {'--use_fp8': None}),
    ('dlrm', 'default', DLRM_UPPER, (200000, 262100, 150000, 300000, 100000), {}),
]

cache_file = sys.argv[1] if len(sys.argv) > 1 else None


def check_cache(name, bs, config_ver, target_qps):
    target_qps = int(target_qps)
    if not cache_file:
        return None
    try:
        with open(cache_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip().split(',')
                result = l[-1]
                fields = l[0:-1]
                fields = (l[0], int(l[1]), l[2], int(l[3]))
                if fields == (name, bs, config_ver, target_qps):
                    if result == 'PASS':
                        return True
                    if result == 'FAIL':
                        return False
    except FileNotFoundError as e:
        pass
    return None


def update_cache(name, bs, config_ver, target_qps, result):
    target_qps = int(target_qps)
    if not cache_file:
        return
    res = "PASS" if result else "FAIL"
    with open(cache_file, 'a') as f:
        s = f"{name},{bs},{config_ver},{int(target_qps)},{res}"
        print("updating cache " + s)
        f.write(s + '\n')


def extra_args_to_string(extra_args):
    s = ""
    for k, v in extra_args.items():
        if v:
            s += k + v + ' '
        else:
            s += k + ' '
    return s


def run_benchmark(target_qps, bs, config_ver, name, extra_args) -> bool:
    target_qps = int(target_qps)
    cache_ret = check_cache(name, bs, config_ver, target_qps)
    if cache_ret is not None:
        print(f"found {name}, bs{bs}, {target_qps} in cache")
        return cache_ret
    duration = 180000  # 3 min
    extra_args_s = extra_args_to_string(extra_args)
    cmd = f"make run_harness RUN_ARGS=\"--benchmarks={name} --scenarios=server --config_ver={config_ver} --gpu_batch_size={bs} --server_target_qps={int(target_qps)} --min_query_count=1 --min_duration={duration} {extra_args_s}\""
    result = sp.run(cmd, capture_output=True, shell=True)
    invalid = 'Performance constraints satisfied : NO' in result.stdout.decode('utf-8')
    ret = result.returncode == 0 and not invalid
    update_cache(name, bs, config_ver, target_qps, ret)
    return ret


def generate_engine(name, bs, config_ver, extra_args) -> bool:
    print(f"[BST] Generating engine for {name} bs {bs} config_ver {config_ver}...")
    extra_args_s = extra_args_to_string(extra_args)
    cmd = f"make generate_engines RUN_ARGS=\"--benchmarks={name} --scenarios=server --config_ver={config_ver} --gpu_batch_size={bs} {extra_args_s}\""
    result = sp.run(cmd, capture_output=True, shell=True)
    if result.returncode != 0:
        print(f"[BST] Error generating engine for {name} bs {bs} config_ver {config_ver}")
        return False
    return True


def bsearch(target_qps_upper, bs, config_ver, name, best_so_far, extra_args):
    target_qps = target_qps_upper
    passed = False
    upper_bound = target_qps_upper
    lower_bound = best_so_far
    print(f"[BST] Running name: {name}, bs: {bs}, config_ver: {config_ver}")
    print(f"[BST] Searching for upper and lower bounds...")
    if not best_so_far:
        while not passed:
            passed = run_benchmark(target_qps, bs, config_ver, name, extra_args)
            if not passed:
                upper_bound = target_qps
                print(f"[BST] failed at {target_qps}qps")
            else:
                print(f"[BST] passed at {target_qps}qps")
                lower_bound = target_qps
                break
            target_qps = target_qps // 2
            target_qps = int(target_qps)
    else:
        passed = run_benchmark(best_so_far, bs, config_ver, name, extra_args)
        if not passed:
            print(f"[BST] aborting, failed at best_so_far {best_so_far}qps")
            return -1
        target_qps = best_so_far
        while passed:
            target_qps *= 2
            passed = run_benchmark(target_qps, bs, config_ver, name, extra_args)
        upper_bound = target_qps

    print(f"[BST] bs: {bs}, uppper_bound: {upper_bound}, lower_bound: {lower_bound}")
    print(f"[BST] finetuning estimate with binary search...")

    if best_so_far is None:
        best_so_far = lower_bound

    ite = 5
    for i in range(ite):
        mid = (upper_bound + lower_bound) // 2
        mid = int(mid)
        passed = run_benchmark(mid, bs, config_ver, name, extra_args)
        if passed:
            print(f"[BST] passed at {mid}qps, increasing lower bound")
            lower_bound = mid
        else:
            print(f"[BST] failed at {mid}qps, reducing upper bound")
            upper_bound = mid
            if best_so_far and upper_bound < best_so_far:
                print(f"[BST] aborting, upper_bound {upper_bound} worse than best_so_far {best_so_far}")
                return -1

    print(f"[BST] best passing qps: {lower_bound}")
    return lower_bound


for benchmark in benchmarks:
    name, config_ver, target_qps_upper, batch_sizes, extra_args = benchmark
    best_so_far = None
    for bs in batch_sizes:
        success = generate_engine(name, bs, config_ver, extra_args)
        if not success:
            print(f"[BST] Failed to build engine, skipping {name} bs {bs}")
            continue
        best_qps = bsearch(target_qps_upper, bs, config_ver, name, best_so_far, extra_args)
        if best_so_far is None:
            best_so_far = best_qps
        else:
            best_so_far = max(best_qps, best_so_far)
    print(f"Best config for {name} {config_ver} is bs {bs} {best_so_far}qps")
