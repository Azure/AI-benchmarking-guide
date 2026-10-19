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
import datetime
import glob
import json
import os
import re
import shutil
import sys
from numbers import Number
from typing import Optional


class SortingCriteria:
    Higher = True
    Lower = False


class SimpleLogger:
    """A basic logger for scripts.
    """

    def __init__(self, indent_size: int = 2, prefix: str = "=>", output_fd=sys.stdout):
        self.indent_size = indent_size
        self.prefix = prefix
        self.output_fd = output_fd
        self.indent_level = 0

    def inc_indent_level(self):
        self.indent_level += 1

    def dec_indent_level(self):
        self.indent_level = max(0, self.indent_level - 1)

    def log(self, *args, **kwargs):
        if len(args) == 0:
            return

        args = list(args)
        args[0] = "  " * self.indent_level + self.prefix + " " + args[0]
        print(*args, file=self.output_fd, **kwargs)


def safe_divide(numerator: Number, denominator: Number) -> Optional[float]:
    """
    Divides 2 numbers, returning None if a DivisionByZero were to occur.

    Args:
        numerator (Number):
            Value for the numerator
        denominator (Number):
            Value for the denominator

    Returns:
        Optional[float]: numerator/denominator. None if denominator is 0.
    """
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def get_system_type(system_name):
    fname = "systems/{:}.json".format(system_name)
    if not os.path.exists(fname):
        raise Exception("Could not locate system.json for {:}".format(system_name))

    with open(fname) as f:
        data = json.load(f)

    if "system_type" not in data:
        raise Exception("{:} does not have 'system_type' key".format(fname))

    return data["system_type"]


def dry_runnable(f):
    """Makes a function "dry-runnable". Adds a new keyword argument 'dry_run: bool = False' to the function.

    If dry_run is set to True, a message is printed instead of running the function.
    """
    def _f(*args, dry_run=False, **kwargs):
        if not dry_run:
            return f(*args, **kwargs)
        else:
            param_str = ", ".join([f"{type(arg).__name__}: {arg}" for arg in args] +
                                  [f"{k}: {type(v).__name__} = {v}" for k, v in kwargs.items()])
            print(f"dry run> {f.__name__}({param_str})")
    return _f


@dry_runnable
def safe_copy(input_file, output_file, logger=None):
    printfn = logger.log if logger is not None else print
    printfn(f"Copy {input_file} -> {output_file}")
    try:
        shutil.copy(input_file, output_file)
    except Exception as e:
        printfn(f"Copy failed. Error: {e}")


@dry_runnable
def safe_copytree(src_dir, dst_dir, logger=None):
    printfn = logger.log if logger is not None else print
    printfn(f"Copy {src_dir} -> {dst_dir}")
    try:
        shutil.rmtree(dst_dir, ignore_errors=True)
        shutil.copytree(src_dir, dst_dir)
    except Exception as e:
        printfn(f"Copytree failed. Error: {e}")


@dry_runnable
def safe_rmtree(dir, logger=None):
    printfn = logger.log if logger is not None else print
    printfn(f"Remove {dir}")
    try:
        shutil.rmtree(dir)
    except Exception as e:
        printfn(f"Rmtree failed. Error: {e}")
