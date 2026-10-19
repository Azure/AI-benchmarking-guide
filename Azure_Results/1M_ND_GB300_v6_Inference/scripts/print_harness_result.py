#! /usr/bin/env python3
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

from nvmitten.configurator import Field
from nvmitten.configurator.fields import parse_fields
from nvmitten.tree import Traversal, Tree
from pathlib import Path
from tabulate import tabulate

import argparse
import glob
import nvmitten.json_utils as json
import os
import sys

# Register JSON objects
from nvmitten.interval import NumericRange
from nvmitten.memory import Memory
from nvmitten.system.component import Description

from code.common.constants import Benchmark, Scenario, WorkloadSetting
from code.common.workload import Workload
from code.common.mlcommons.accuracy_checker import check_accuracy


def enumerate_results(base_dir: Path):
    t = Tree("results", None)

    # Find all metadata.json files
    md_paths = glob.glob(str(base_dir / "**" / "metadata.json"), recursive=True)
    for md_path in md_paths:
        with open(md_path) as f:
            _dat = json.load(f)

        system_name = _dat["system_name"]
        benchmark_name = _dat["benchmark_full"]
        scenario = _dat["scenario"]
        workload_setting = _dat["workload_setting_code"]
        test_mode = _dat["test_mode"][:-4].lower()

        filter_keys = ["system_name",
                       "benchmark_full",
                       "workload_setting_code",
                       "result_validity",
                       "effective_min_duration_ms",
                       "scenario_key",
                       "true_result_value",
                       "true_result_metric",
                       "power_meter_enabled",
                       "avg_power",
                       "dlrm_pairs_per_second"]

        if test_mode == "accuracy":
            filter_keys.extend(["accuracy_pass", "accuracy_raw"])
            if "accuracy_raw" not in _dat:
                # The accuracy checker has not yet run. Reconstruct Workload and run it.
                wl = Workload(Benchmark.get_match(_dat["benchmark_short"]),
                              Scenario.get_match(_dat["scenario"]),
                              setting=WorkloadSetting.from_short(_dat["workload_setting_code"]),
                              log_dir=_dat["base_log_dir"])
                acc_results = check_accuracy(wl)
                summary_strings = []
                acc_targets = []
                final_acc_result = True
                for acc_res in acc_results:
                    pass_string = "PASSED"
                    if not acc_res["pass"]:
                        pass_string = "FAILED"
                        final_acc_result = False
                    name, val, thresh = (acc_res["name"], acc_res["value"], acc_res["threshold"])
                    if "upper_limit" in acc_res:
                        ul = acc_res["upper_limit"]
                        acc_targets.append((name, val, thresh, ul))
                        summary_strings.append(f"[{pass_string}] {name}: {val:.3f} (Valid Range=[{thresh:.3f}, {ul:.3f}])")
                    else:
                        acc_targets.append((name, val, thresh))
                        summary_strings.append(f"[{pass_string}] {name}: {val:.3f} (Threshold={thresh:.3f})")
                _dat["accuracy"] = acc_results
                _dat["accuracy_pass"] = final_acc_result
                _dat["accuracy_raw"] = tuple(acc_targets)
                _dat["summary_string"] = " | ".join(summary_strings)
                with open(md_path, "w") as f:
                    json.dump(_dat, f, indent=4, sort_keys=True)

        if scenario.lower() == "server":
            filter_keys.extend(["latency_usage_ttft",
                                "latency_usage_tpot",
                                "latency_usage_raw"])
        keyspace = [test_mode, scenario, system_name, benchmark_name, workload_setting]
        t[keyspace] = {k: v for k, v in _dat.items() if k in filter_keys}
    return t


def print_session_results(base_dir: Path) -> bool:
    # TODO: Add regression metrics when result logs are collected into the artifacts repo.
    results = enumerate_results(base_dir)

    print(f"\n{'='*24} Result summaries: {'='*24}\n")
    all_acc_pass = True
    for test_mode_node in results.get_children():
        test_mode = test_mode_node.name

        for scenario_node in test_mode_node.get_children():
            header = ["System Name", "Benchmark", "Setting", "Valid?"]

            if test_mode == "accuracy":
                header.pop(-1)
                header.extend(["All Acc. Pass?", "Metric Name", "Measured Value", "Threshold"])
            else:
                if scenario_node.name.lower() == "server":
                    header.append("Per-query time usage")

                header.extend(["Metric Name", "Measured Value", "Avg. Power (W)"])

            print(f"{scenario_node.name} Scenario:")
            table = list()
            for node in scenario_node.traversal(order=Traversal.OnlyLeaves):
                dat = node.value
                if test_mode == "accuracy":
                    for i, _tup in enumerate(dat["accuracy_raw"]):
                        thresh_string = ">=" + str(_tup[2])
                        if len(_tup) == 4:
                            thresh_string += ", <=" + str(_tup[3])

                        if i == 0:
                            if not dat["accuracy_pass"]:
                                all_acc_pass = False

                            row = [dat["system_name"],
                                   dat["benchmark_full"],
                                   dat["workload_setting_code"],
                                   "Yes" if dat["accuracy_pass"] else "No",
                                   _tup[0],
                                   _tup[1],
                                   thresh_string]
                        else:
                            row = ([""] * 4) + [_tup[0], _tup[1], thresh_string]
                        table.append(tuple(row))
                else:
                    min_duration_satisfied = dat["effective_min_duration_ms"] >= 60 * 10 * 1000
                    validity = dat["result_validity"] if min_duration_satisfied else "INVALID (duration)"
                    row = [dat["system_name"],
                           dat["benchmark_full"],
                           dat["workload_setting_code"],
                           validity]

                    if scenario_node.name.lower() == "server":
                        ttft_ratio = dat.get("latency_usage_ttft", 0.0) * 100
                        tpot_ratio = dat.get("latency_usage_tpot", 0.0) * 100
                        serv_ratio = dat.get("latency_usage_raw", 0.0) * 100

                        if ttft_ratio:
                            row.append(f"TTFT: {ttft_ratio:.1f}%, TPOT: {tpot_ratio:.1f}")
                        else:
                            row.append(f"{serv_ratio:.1f}%")

                    avg_power = dat.get("avg_power", None)
                    row.extend([dat["true_result_metric"],
                                dat["true_result_value"],
                                avg_power if avg_power else "N/A"])

                    table.append(tuple(row))

                    if "dlrm_pairs_per_second" in dat:
                        fill_count = 5 if scenario_node.name.lower() == "server" else 4
                        table.append(([""] * fill_count) + ["dlrm_pairs_per_second", dat["dlrm_pairs_per_second"], ""])
            print(tabulate(table,
                           headers=header,
                           tablefmt="outline",
                           floatfmt=".2f"))
            if scenario_node.name.lower() == "server":
                print("  * Note: 'Per-query time usage' is the measured 99th-percentile latency divided by the"
                      " requested server latency. This value should not exceed 100% for a 'VALID' result.")
    return all_acc_pass


if __name__ == "__main__":
    _f = Field(
        "log_dir",
        description="Directory for all output logs.",
        from_string=Path)

    log_dir = parse_fields([_f]).get("log_dir", os.environ.get("LOG_DIR", None))
    if not log_dir:
        print("No log_dir specified")
    else:
        all_acc_pass = print_session_results(log_dir)
        if all_acc_pass:
            sys.exit(0)
        else:
            sys.exit(1)
