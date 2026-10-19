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


__doc__ = """NVIDIA's MLPerf Inference Benchmark submission code. NVIDIA's implementation runs in 2 phases.

The first phase is 'engine generation', which builds a TensorRT Engine using TensorRT, a Deep Learning Inference
performance optimization SDK by NVIDIA. This only applies to NVIDIA accelerator-based workloads.

The second phase is a 'harness run', which launches the generated TensorRT engine in a server-like harness that
accepts input from LoadGen (MLPerf Inference's official Load Generator), runs the inference with the engine, and reports
the output back to LoadGen.

More about the MLPerf Inference Benchmark and NVIDIA's submission implementation can be found in the README.md for this
project.
"""
from code import G_BENCHMARK_MODULES
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple

from nvmitten.configurator import (
    Configuration,
    ConfigurationIndex,
    Field,
    HelpInfo,
    autoconfigure,
    bind,
)
from nvmitten.importer import ScopedImporter
from nvmitten.pipeline import Pipeline, ScratchSpace
from nvmitten.system.system import System

import code.common.constants as C
import code.common.paths as paths
import code.fields.gen_engines as builder_fields
import code.fields.harness as harness_fields
import code.fields.meta as metafields
import code.fields.general as general_fields
import code.ops as Ops

from code.common import logging
from code.common.power_limit import get_power_context
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.utils import prepare_virtual_env
from code.common.workload import Workload
from code.common.mlcommons.compliance import get_audit_verifier, set_audit_conf
from code.llmlib.config import HarnessConfig, TrtllmEndpointConfig, TrtllmHlApiConfig


@autoconfigure
@bind(metafields.action)
@bind(metafields.benchmarks)
@bind(metafields.scenarios)
@bind(metafields.harness_type)
@bind(metafields.accuracy_target)
@bind(metafields.power_setting)
@bind(general_fields.config_dir)
@bind(harness_fields.audit_test)
@bind(general_fields.show_help, "show_help")
class MainRunner:
    def __init__(self,
                 system: System,
                 action: C.Action = None,
                 benchmarks: List[C.Benchmark] = None,
                 scenarios: List[C.Scenario] = None,
                 harness_type: Optional[C.HarnessType] = None,
                 accuracy_target: C.AccuracyTarget = C.AccuracyTarget(0.99),
                 power_setting: C.PowerSetting = C.PowerSetting.MaxP,
                 show_help: bool = False,
                 config_dir: os.PathLike = paths.WORKING_DIR / "configs",
                 audit_test: Optional[C.AuditTest] = None):
        assert action is not None, "No action specified"
        assert benchmarks is not None, "No benchmarks specified"
        assert scenarios is not None, "No scenarios specified"

        self.system = system
        self.system_id = system.extras["id"]
        self.action = action
        self.benchmarks = benchmarks
        self.scenarios = scenarios

        self.harness_type = harness_type
        self.accuracy_target = accuracy_target
        self.power_setting = power_setting

        self.config_dir = config_dir
        self.audit_test = audit_test

        self.show_help = show_help

        self.config_index = ConfigurationIndex()
        for benchmark in self.benchmarks:
            with prepare_virtual_env(benchmark):
                for scenario in self.scenarios:
                    p = self.conf_path(benchmark, scenario)
                    if not p.exists():
                        logging.info(f"Config file {p} not found. Loading minimal configs as default.")
                        _subdir = "minimal"
                    else:
                        logging.info(f"Loading configs from {p}")
                        _subdir = self.system_id

                    with ScopedImporter([str(self.config_dir / _subdir)] + sys.path):
                        imp_path = f"{scenario.valstr}.{benchmark.valstr}"
                        self.config_index.load_module(imp_path, prefix=[_subdir, benchmark, scenario])

    def conf_path(self, benchmark: C.Benchmark, scenario: C.Scenario) -> Path:
        """Get the path to the configuration file for a given benchmark and scenario.

        Args:
            benchmark (C.Benchmark): The benchmark to get the config path for
            scenario (C.Scenario): The scenario to get the config path for

        Returns:
            Path: The path to the configuration file
        """
        return self.config_dir / self.system_id / scenario.valstr / f"{benchmark.valstr}.py"

    def _run_workload(self, benchmark: C.Benchmark, scenario: C.Scenario):
        """Run a specific workload for a given benchmark and scenario.

        This method sets up the workload configuration, creates a pipeline, and executes it
        with the appropriate power context.

        Args:
            benchmark (C.Benchmark): The benchmark to run
            scenario (C.Scenario): The scenario to run the benchmark under
        """
        # Override log directory for audit tests
        if self.audit_test is not None:
            # Check if we even need to run the audit test in the first place
            verifier = get_audit_verifier(self.audit_test)
            if benchmark in verifier.exclude_list:
                logging.info(f"Skipping audit test {self.audit_test.valstr} for {benchmark.valstr} {scenario.valstr} as it is not needed for submission.")
                return

            _audit_log_dir = paths.BUILD_DIR / "compliance_logs" / self.audit_test.valstr
            _audit_log_dir.mkdir(parents=True, exist_ok=True)
            os.environ["LOG_DIR"] = str(_audit_log_dir)

        # self.audit_test == None will cleanup any audit configs before harness starts
        set_audit_conf(self.audit_test, benchmark)

        ht = benchmark.default_harness_type if self.harness_type is None else self.harness_type
        workload_setting = C.WorkloadSetting(harness_type=ht,
                                             accuracy_target=self.accuracy_target,
                                             power_setting=self.power_setting)
        keyspace = [self.system_id, benchmark, scenario, workload_setting]
        config = self.config_index.get(keyspace)
        if config is None:
            logging.warning(f"Config not found for current system. Attempting to load minimal config for {benchmark.valstr} {scenario.valstr} ({workload_setting.short})")
            config = self.config_index.get(["minimal", benchmark, scenario, workload_setting])
            if config is None:
                logging.error("No minimal config found. Using empty config")
                config = Configuration()

        # Sanitize configuration for types. Maybe this should be provided in Mitten?
        for k, v in config.items():
            assert isinstance(k, Field), f"Invalid Configuration key {k} is not a Mitten Field object"
            if isinstance(v, str) and (k.from_string and k.from_string is not str):
                logging.debug(f"Configuration - Parsing string for field {k.name}")
                config[k] = k.from_string(v)

        # Use .from_fields since there is no auto-applied config yet.
        wl = Workload.from_fields(benchmark,
                                  scenario,
                                  system=self.system,
                                  setting=workload_setting)
        config[Workload.FIELD] = wl

        if self.action == C.Action.GenerateEngines:
            config[builder_fields.force_build_engines] = True

        power_context = get_power_context()

        with config.autoapply():
            if benchmark.is_llm and benchmark is not C.Benchmark.WHISPER:  # Whisper is using PyHarnessOp in run harness
                core_type = HarnessConfig().core_type

                if self.action in (C.Action.GenerateTritonConfig,):
                    ops = self.get_triton_generate_config_op(benchmark)

                if self.action in (C.Action.GenerateDisaggConfig,):
                    ops = self.get_trtllm_disagg_generate_config_op(benchmark)

                if self.action in (C.Action.GenerateEngines,):
                    ops = self.get_llm_generate_engine_ops(benchmark, core_type)

                if self.action in (C.Action.RunLLMServer,):
                    ops = self.get_llm_launch_server_ops(benchmark, core_type)

                if self.action in (C.Action.RunHarness,):
                    ops = self.get_llm_harness_run_ops(benchmark, core_type)

            else:
                ops = self.get_harness_run_ops(benchmark)

            if self.show_help:
                print(HelpInfo.build_help_string(ops))
                sys.exit(0)

            scratch_space = ScratchSpace(paths.BUILD_DIR)
            pipeline = Pipeline(scratch_space, ops, dict())
            with power_context:
                pipeline.run()

    def get_llm_generate_engine_ops(self, benchmark: C.Benchmark, core_type: Optional[harness_fields.CoreType] = None):
        """
        Get list of operations to generate LLM engines for given --core_type
        This will build one or more engines as needed by the workload.
        """

        def get_build_ops(pipeline: Tuple[str] = ()):
            m = G_BENCHMARK_MODULES[benchmark]
            m.load(pipeline)

            impls = m.custom_op_impls
            ops = []

            for k in list(pipeline):
                if impls[k] is not None:
                    ops.append(impls[k])

            if self.show_help and impls["EngineBuilderOp"]:
                for builder in m.component_map.values():
                    if builder:
                        for c in builder.mro():
                            HelpInfo.add_configurator_dependency(impls["EngineBuilderOp"], c)

            return ops

        match core_type:
            case harness_fields.CoreType.TRITON_GRPC: ops = get_build_ops(("CalibrateEngineOp", "EngineBuilderOp", "GenerateTritonConfigOp",))
            case harness_fields.CoreType.TRTLLM_EXECUTOR: ops = get_build_ops(("CalibrateEngineOp", "EngineBuilderOp", ))
            case harness_fields.CoreType.TRTLLM_DISAGG: ops = get_build_ops(("HFQuantizerOp", ))
            case harness_fields.CoreType.TRTLLM_ENDPOINT:
                requires_engine = TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'cpp'
                ops = get_build_ops(("HFQuantizerOp",) if not requires_engine else ("CalibrateEngineOp", "EngineBuilderOp", ))
            case harness_fields.CoreType.TRTLLM_HLAPI:
                requires_engine = TrtllmHlApiConfig().runtime_flags['trtllm_backend'] == 'cpp'
                ops = get_build_ops(("HFQuantizerOp",) if not requires_engine else ("CalibrateEngineOp", "EngineBuilderOp", ))
            case _: raise NotImplementedError(f"Unsupported core type: {core_type}")
        return ops

    def get_llm_launch_server_ops(self, benchmark: C.Benchmark, core_type: Optional[harness_fields.CoreType] = None):
        """
        Get list of operations to launch LLM Servers for given core_type
        This will launch one or multiple LLM Servers to expose benchmark-able endpoints.
        This will also run calibration and generate engines if needed.
        """

        def get_launch_ops(additional_ops: Tuple[str] = ()):
            ops = self.get_llm_generate_engine_ops(benchmark, core_type)

            m = G_BENCHMARK_MODULES[benchmark]
            m.load(additional_ops)
            for k in additional_ops:
                if m.custom_op_impls[k] is not None:
                    ops.append(m.custom_op_impls[k])

            return ops

        match core_type:
            case harness_fields.CoreType.TRITON_GRPC: ops = get_launch_ops(("GenerateTritonConfigOp", "RunTritonServerOp"))
            case harness_fields.CoreType.TRTLLM_ENDPOINT: ops = self.get_llm_generate_engine_ops(benchmark, core_type) + get_launch_ops(("RunTrtllmServeOp",))
            case harness_fields.CoreType.TRTLLM_DISAGG: ops = get_launch_ops(("RunTrtllmServeDisaggOp",))
            case harness_fields.CoreType.TRTLLM_EXECUTOR: ops = []  # no server
            case harness_fields.CoreType.TRTLLM_HLAPI: ops = []  # no server
            case _: raise NotImplementedError(f"Unsupported core type: {core_type}")
        return ops

    def get_llm_harness_run_ops(self, benchmark: C.Benchmark, core_type: Optional[harness_fields.CoreType] = None):
        """Get the list of operations to be performed for a given llm benchmark.

        The operations list varies depending on the core_type being used.

        Args:
            benchmark (C.Benchmark): The benchmark to get operations for
        """
        def get_run_ops(backend_modules: Tuple[str] = ()):
            m = G_BENCHMARK_MODULES[benchmark]
            m.load(backend_modules)
            impls = {
                "LoadgenConfFilesOp": Ops.LoadgenConfFilesOp,
                "ResultSummaryOp": Ops.ResultSummaryOp,
            }
            impls |= m.custom_op_impls

            ops = []
            pipeline_steps = ["LoadgenConfFilesOp"] + list(backend_modules) + ["ResultSummaryOp"]
            for k in pipeline_steps:
                if impls[k] is not None:
                    ops.append(impls[k])

            return ops

        match core_type:
            case harness_fields.CoreType.TRTLLM_EXECUTOR: ops = get_run_ops(("TrtllmExecutorBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRTLLM_ENDPOINT: ops = get_run_ops(("TrtllmServeBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRTLLM_DISAGG: ops = get_run_ops(("TrtllmDisaggServeBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRITON_GRPC: ops = get_run_ops(("TritonBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRTLLM_HLAPI: ops = get_run_ops(("TrtllmHLApiBenchmarkHarnessOp",))
            case _: raise NotImplementedError(f"Unsupported core type: {core_type}")

        # NOTE(vir):
        # add engine generation steps in pipeline to satisfy dependency outputs
        # engine build ops do not overwrite existing engines by default
        ops = self.get_llm_generate_engine_ops(benchmark, core_type) + ops
        return ops

    def get_harness_run_ops(self, benchmark: C.Benchmark):
        """Get the list of operations to be performed for a given benchmark.

        The operations list varies depending on the action type (GenerateEngines or RunHarness)
        and includes operations like calibration, engine building, and result summarization.

        Args:
            benchmark (C.Benchmark): The benchmark to get operations for

        Returns:
            list: List of operations to be performed for the benchmark
        """
        m = G_BENCHMARK_MODULES[benchmark]
        m.load()
        impls = {"CalibrateEngineOp": Ops.CalibrateEngineOp,
                 "EngineBuilderOp": Ops.EngineBuilderOp,
                 "BenchmarkHarnessOp": Ops.BenchmarkHarnessOp,
                 "ResultSummaryOp": Ops.ResultSummaryOp}
        impls |= m.custom_op_impls

        ops = []
        for k in ("CalibrateEngineOp", "EngineBuilderOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        if self.show_help and impls["EngineBuilderOp"]:
            for builder in m.component_map.values():
                if builder:
                    for c in builder.mro():
                        HelpInfo.add_configurator_dependency(impls["EngineBuilderOp"], c)

        if self.action == C.Action.GenerateEngines:
            return ops

        ops.append(Ops.LoadgenConfFilesOp)
        for k in ("BenchmarkHarnessOp", "ResultSummaryOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        return ops

    def get_triton_generate_config_op(self, benchmark: C.Benchmark):
        """Get the list of operations to generate Triton Server config files.
        """
        m = G_BENCHMARK_MODULES[benchmark]
        m.load(("GenerateTritonConfigOp", ))
        impls = {"CalibrateEngineOp": Ops.CalibrateEngineOp,
                 "EngineBuilderOp": Ops.EngineBuilderOp}
        impls |= m.custom_op_impls

        ops = []

        for k in ("CalibrateEngineOp", "EngineBuilderOp", "GenerateTritonConfigOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        return ops

    def get_trtllm_disagg_generate_config_op(self, benchmark: C.Benchmark):
        """Get the list of operations to generate Trtllm Disagg Server config files.
        """
        m = G_BENCHMARK_MODULES[benchmark]
        m.load(("HFQuantizerOp", "GenerateTrtllmDisaggConfigOp",))
        impls = m.custom_op_impls
        ops = []

        for k in ("HFQuantizerOp", "GenerateTrtllmDisaggConfigOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        return ops

    def run_all(self):
        """Run all configured workloads.

        This method iterates through all configured benchmarks and scenarios,
        running each workload in sequence.
        """
        for benchmark in self.benchmarks:
            with prepare_virtual_env(benchmark):
                for scenario in self.scenarios:
                    self._run_workload(benchmark, scenario)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    Ops.MPS().disable()
    if "id" not in DETECTED_SYSTEM.extras:
        logging.info(f"Detected system did not match any known systems. Exiting. {DETECTED_SYSTEM}")
    else:
        logging.info(f"Detected system ID: {DETECTED_SYSTEM.extras['id']}")
        with Configuration().autoapply():  # Create empty Configuration to invoke autoconfigure
            runner = MainRunner(DETECTED_SYSTEM)
        runner.run_all()
