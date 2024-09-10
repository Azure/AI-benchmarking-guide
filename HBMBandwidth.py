import json
import os
import statistics
import subprocess
import time
import csv
from prettytable import PrettyTable
import numpy as np
import torch

from Infra import sqlite

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from matplotlib.ticker import


class HBMBandwidth:
    def __init__(self, path: str, machine: str):
        self.name = "HBMBandwidth"
        self.machine_name = machine
        config = self.get_config(path)
        self.num_runs, self.interval = self.config_conversion(config)

        self.buffer = []

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def parse_json(self, config):
        return config["inputs"]["num_runs"], config["inputs"]["interval"]

    def config_conversion(self, config) -> tuple[list, list, list]:
        return self.parse_json(config)

    def build(self):
        current = os.getcwd()

        path = "BabelStream"
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(
                ["git", "clone", "https://github.com/gitaumark/BabelStream",  path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        build_path = os.path.join(current, "BabelStream")
        os.chdir(build_path)

        babelstream_build_path = os.path.join(build_path, "build")
        if not os.path.isdir(babelstream_build_path):
            os.mkdir(babelstream_build_path)
            os.chdir(babelstream_build_path)
            results = subprocess.run(
                [
                    "cmake",
                    "../",
                    "-DMODEL=cuda",
                    "-DCUDA_ARCH=sm_90",
                    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(results.stderr.decode('utf-8')) 
            results = subprocess.run(
                ["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(results.stderr.decode('utf-8')) 
        else:
            os.chdir(babelstream_build_path)


    def run(self):
        current = os.getcwd()
        print("Running HBM Bandwidth...")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        runs_executed = 0
        buffer = []
        while runs_executed < self.num_runs:
            start.record()
            results = subprocess.run(
                ["./cuda-stream"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            end.record()
            torch.cuda.synchronize()
            log = results.stdout.decode("utf-8").strip().split("\n")[14:19]
            for i in range(len(log)):
                temp = log[i].split()
                log[i] = [temp[0], temp[1]]
            buffer.append(log)

            runs_executed += 1
            time.sleep(int(self.interval))

    
        self.buffer = buffer
        os.chdir(current)
        self.save_results()

    def process_stats(self, results):
        mean = statistics.mean(results)/1000000
        maximum = max(results)/1000000
        minimum = min(results)/1000000
        stdev = statistics.stdev(results)/1000
        return [minimum, maximum, mean, stdev]
    

    def save_results(self):
        copy = ["Copy"]
        mul = ["Mul"]
        add = ["Add"]
        triad = ["Triad"]
        dot = ["Dot"]
        for log in self.buffer:
            copy.append(float(log[0][1]))
            mul.append(float(log[1][1]))
            add.append(float(log[2][1]))
            triad.append(float(log[3][1]))
            dot.append(float(log[4][1]))

        copy[1:] = self.process_stats(copy[1:])
        mul[1:] = self.process_stats(mul[1:])
        add[1:] = self.process_stats(add[1:])
        triad[1:] = self.process_stats(triad[1:])
        dot[1:] = self.process_stats(dot[1:])
        
        table1 = PrettyTable()
        table1.field_names = ["Operation","Min (TB/s)", "Max (TB/s)", "Mean (TB/s)", "StDev (GB/s)"]
        table1.add_row(copy)
        table1.add_row(mul)
        table1.add_row(add)
        table1.add_row(triad)
        table1.add_row(dot)
        print(table1)

        with open('../../Outputs/HBMBandwidth_Performance_results_' + self.machine_name +'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Operation","Min (TB/s)", "Max (TB/s)", "Mean (TB/s)", "StDev (GB/s)"])
            writer.writerow(copy)
            writer.writerow(mul)
            writer.writerow(add)
            writer.writerow(triad)
            writer.writerow(dot)
