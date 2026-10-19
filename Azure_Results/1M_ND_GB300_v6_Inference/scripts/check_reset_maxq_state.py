# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from code.common import run_command
import argparse

if __name__ == "__main__":
    # Set default target Orin MaxP Gpu/EMC/Dla Core/Dla Falcon frequency here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        help="Specifies the Orin MaxP Target GPU Frequency.",
        default="1300500000",
        type=str
    )
    parser.add_argument(
        "--emc",
        help="Specifies the Orin MaxP Target EMC Frequency.",
        default="3199000000",
        type=str
    )
    parser.add_argument(
        "--dla_core",
        help="Specifies the Orin MaxP Target DLA Core Frequency.",
        default="1600000000",
        type=str
    )
    parser.add_argument(
        "--dla_falcon",
        help="Specifies the Orin MaxP Target DLA Falcon Frequency.",
        default="844800000",
        type=str
    )
    args = parser.parse_args()

    gpu_freq = run_command("sudo cat /sys/kernel/debug/bpmp/debug/clk/gpusysclk/rate", get_output=True, tee=False)
    emc_freq = run_command("sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate", get_output=True, tee=False)
    dla0_freq = run_command("sudo cat /sys/kernel/debug/bpmp/debug/clk/nafll_dla0_core/rate", get_output=True, tee=False)
    dla1_freq = run_command("sudo cat /sys/kernel/debug/bpmp/debug/clk/nafll_dla1_core/rate", get_output=True, tee=False)
    dla0_falcon_freq = run_command("sudo cat /sys/kernel/debug/bpmp/debug/clk/nafll_dla0_falcon/rate", get_output=True, tee=False)
    dla1_falcon_freq = run_command("sudo cat /sys/kernel/debug/bpmp/debug/clk/nafll_dla1_falcon/rate", get_output=True, tee=False)

    if not args.gpu in gpu_freq:
        print("GPU frequency not set to maximum")
        exit(1)
    if not args.emc in emc_freq:
        print("EMC frequency not set to maximum")
        exit(1)
    if not args.dla_core in dla0_freq:
        print("DLA0_CORE frequency not set to maximum")
        exit(1)
    if not args.dla_core in dla1_freq:
        print("DLA1_CORE frequency not set to maximum")
        exit(1)
    if not args.dla_falcon in dla0_falcon_freq:
        print("DLA0_FALCON frequency not set to maximum")
        exit(1)
    if not args.dla_falcon in dla1_falcon_freq:
        print("DLA1_FALCON frequency not set to maximum")
        exit(1)
    print("MaxP Target Frequency Check Passed!")
