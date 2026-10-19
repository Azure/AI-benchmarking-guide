#!/usr/bin/env python3
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

from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple
from code.common.constants import Benchmark


_required_op_impls = ("CalibrateEngineOp",
                      "EngineBuilderOp",
                      "BenchmarkHarnessOp")


class ModuleLocation:
    def __init__(self, path: Path):
        self.path = path
        self._m = None
        self.custom_op_impls = {}

    def load(self, extra_op_impls: Tuple[str, ...] = ()):
        op_impls = extra_op_impls + _required_op_impls
        if not self._m:
            self._m = import_module(self.path)
        for op_impl in op_impls:
            if hasattr(self._m, op_impl):
                self.custom_op_impls[op_impl] = getattr(self._m, op_impl)
        return self._m

    @property
    def component_type(self):
        return self.load().Component

    @property
    def component_map(self):
        return self.load().COMPONENT_MAP

    @property
    def valid_component_sets(self):
        return self.load().VALID_COMPONENT_SETS


# Instead of storing the objects themselves in maps, we store object locations, as we do not want to import redundant
# modules on every run. Some modules include CDLLs and TensorRT plugins, or have large imports that impact runtime.
# Dynamic imports are also preferred, as some modules (due to their legacy model / directory names) include dashes.
#
# These are stored here, rather than being a property of Benchmark itself, so that other submission divisions can more
# easily remap where their benchmark code is stored, which still using the same import paths as the closed division.
G_BENCHMARK_MODULES: Dict[Benchmark, ModuleLocation] = {
    Benchmark.ResNet50: ModuleLocation("code.resnet50.tensorrt"),
    Benchmark.Retinanet: ModuleLocation("code.retinanet.tensorrt"),
    Benchmark.BERT: ModuleLocation("code.bert.tensorrt"),
    Benchmark.DLRMv2: ModuleLocation("code.dlrm-v2.tensorrt"),
    Benchmark.GPTJ: ModuleLocation("code.gptj.tensorrt"),
    Benchmark.LLAMA2: ModuleLocation("code.llama2-70b.tensorrt"),
    Benchmark.LLAMA3_1_8B: ModuleLocation("code.llama3_1-8b.tensorrt"),
    Benchmark.LLAMA3_1_405B: ModuleLocation("code.llama3_1-405b.tensorrt"),
    Benchmark.Mixtral8x7B: ModuleLocation("code.mixtral-8x7b.tensorrt"),
    Benchmark.DeepSeek_R1: ModuleLocation("code.deepseek-r1.tensorrt"),
    Benchmark.SDXL: ModuleLocation("code.stable-diffusion-xl.tensorrt"),
    Benchmark.RGAT: ModuleLocation("code.rgat.pytorch"),
    Benchmark.WHISPER: ModuleLocation("code.whisper.tensorrt"),
}
