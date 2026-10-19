# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
set -x

# nvidia-smi -lgc 1965" ### This way doesn't work on GB200-NVL. https://nvbugspro.nvidia.com/bug/5069775

hostname
nvidia-smi

MAX_GPU_CLOCK=$(nvidia-smi -q -d CLOCK | grep -m 1 -A 1 Max | awk '/Graphics/ {print $3}')
MAX_MEM_CLOCK=$(nvidia-smi -q -d CLOCK | grep -m 1 -A 4 Max | awk '/Memory/ {print $3}')

echo "Setting application clock to Mem Clock: $MAX_MEM_CLOCK and GPU Clock: $MAX_GPU_CLOCK."

sudo nvidia-smi -rgc
sudo nvidia-smi -ac $MAX_MEM_CLOCK,$MAX_GPU_CLOCK
