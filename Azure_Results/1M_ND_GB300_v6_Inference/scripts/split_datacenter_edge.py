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

import argparse
import copy
import glob
import json
import os
import shutil

from scripts.utils import get_system_type


def safe_copy(input_file, output_file, dry_run=False):
    print("Copy {:} -> {:}".format(input_file, output_file))
    if not dry_run:
        try:
            shutil.copy(input_file, output_file)
        except Exception as e:
            print("Copy failed. Error: {:}".format(e))


def safe_copytree(src_dir, dst_dir, dry_run=False):
    print("Copy {:} -> {:}".format(src_dir, dst_dir))
    if not dry_run:
        try:
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(src_dir, dst_dir)
        except Exception as e:
            print("Copytree failed. Error: {:}".format(e))


def find_multitype_systems(system_desc_dir="systems"):
    """
    Returns a list of system IDs that exist in the system description directory that are marked as both datacenter and
    edge system types.
    """
    all_system_jsons = glob.glob(os.path.join(system_desc_dir, "*.json"))
    multitype_systems = []
    for sys_json in all_system_jsons:
        sys_id = os.path.basename(sys_json)[:-len(".json")]
        sys_type = get_system_type(sys_id)
        if sys_type == "datacenter,edge":
            multitype_systems.append(sys_id)
    return multitype_systems


def split_dir_by_system_type(basedir):
    """
    Given basedir in the form '<parent dir>/<system id>', split it into '<parent dir>/<system id>_datacenter' and
    '<parent dir>/<system_id>_edge'.
    """
    datacenter_basedir = basedir + "_datacenter"
    edge_basedir = basedir + "_edge"

    # Create new directories
    os.makedirs(datacenter_basedir, exist_ok=True)
    os.makedirs(edge_basedir, exist_ok=True)

    for benchmark in os.listdir(basedir):
        benchmark_dir = os.path.join(basedir, benchmark)
        if not os.path.isdir(benchmark_dir):
            print("Warning: {} is not a directory".format(benchmark_dir))
            continue

        os.makedirs(os.path.join(datacenter_basedir, benchmark), exist_ok=True)
        if benchmark not in ["bert-99.9", "dlrm-99", "dlrm-99.9"]:
            os.makedirs(os.path.join(edge_basedir, benchmark), exist_ok=True)

        for scenario in os.listdir(benchmark_dir):
            scenario_dir = os.path.join(benchmark_dir, scenario)
            if not os.path.isdir(scenario_dir):
                print("Warning: {} is not a directory".format(benchmark_dir))
                continue

            # Move scenarios to their appropriate system type, skip if that system type does not support the benchmark
            if scenario in ["Offline", "Server"]:
                safe_copytree(scenario_dir, os.path.join(datacenter_basedir, benchmark, scenario), dry_run=False)

            if benchmark not in ["bert-99.9", "dlrm-99", "dlrm-99.9"]:
                if scenario in ["Offline", "SingleStream", "MultiStream"]:
                    safe_copytree(scenario_dir, os.path.join(edge_basedir, benchmark, scenario), dry_run=False)

    # Delete original directory
    shutil.rmtree(basedir)


def split_system_desc(sys_id):
    """
    Given a system ID that is marked as both datacenter and edge, create 2 new system IDs to denote the individual
    datacenter and edge systems and split the relevant submission files. This method will:

        1. Create 2 new entries in systems/, deleting the old one.
        2. Split the entries in the following directories into the corresponding edge and datacenter scenarios. Offline
           will be duplicated between the two.
            - compliance/
            - measurements/
            - results/
           The original directories will be removed.
    """
    with open(os.path.join("systems", sys_id + ".json")) as f:
        original_dict = json.loads(f.read())
    assert original_dict["system_type"] == "datacenter,edge"

    # Create datacenter and edge system.jsons
    datacenter_dict = copy.deepcopy(original_dict)
    datacenter_dict["system_type"] = "datacenter"
    with open(os.path.join("systems", sys_id + "_datacenter.json"), 'w') as f:
        json.dump(datacenter_dict, f, indent=4, sort_keys=True)

    edge_dict = copy.deepcopy(original_dict)
    edge_dict["system_type"] = "edge"
    with open(os.path.join("systems", sys_id + "_edge.json"), 'w') as f:
        json.dump(edge_dict, f, indent=4, sort_keys=True)

    # Delete the original system description
    os.remove(os.path.join("systems", sys_id + ".json"))

    # Split compliance/
    split_dir_by_system_type(os.path.join("compliance", sys_id))

    # Split measurements/
    split_dir_by_system_type(os.path.join("measurements", sys_id))

    # Split results/
    split_dir_by_system_type(os.path.join("results", sys_id))


if __name__ == "__main__":
    for sys_id in find_multitype_systems():
        split_system_desc(sys_id)
