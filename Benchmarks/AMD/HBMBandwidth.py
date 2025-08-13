import json
import os
import statistics
import time
import csv
from prettytable import PrettyTable
import subprocess
from Infra import tools

class HBMBandwidth:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "HBMBandwidth"
        self.machine_name = machine
        config = self.get_config(dir_path + '/' + config_path)
        self.num_runs, self.interval = self.config_conversion(config)
        self.dir_path = dir_path
        self.container = None
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
        path = "BabelStream"
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = "git clone https://github.com/gitaumark/BabelStream " + self.dir_path + "/BabelStream"
            results = subprocess.run(clone_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            results = subprocess.run('cd ' + self.dir_path + '/BabelStream && cmake -Bbuild -H. -DMODEL=hip -DRELEASE_FLAGS="-O3" -DCMAKE_CXX_COMPILER=hipcc && cmake --build build', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))

    def run(self):
        print("Running HBM Bandwidth...")
        runs_executed = 0
        buffer = []
        while runs_executed < self.num_runs:
            run_cmd = "sudo " + self.dir_path + "/BabelStream/build/hip-stream"
            results = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            log = results.stdout.decode("utf-8").strip().split("\n")[13:18]
            for i in range(len(log)):
                temp = log[i].split()
                log[i] = [temp[0], temp[1]]
            buffer.append(log)

            runs_executed += 1
            time.sleep(int(self.interval))

        self.buffer = buffer
        self.save_results()

    def process_stats(self, results):
        mean = statistics.mean(results)/1000000
        maximum = max(results)/1000000
        minimum = min(results)/1000000
        stdev = statistics.stdev(results)/1000
        return [round(minimum, 2), round(maximum, 2), round(mean, 2)]

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
        table1.field_names = ["Operation","Min (TB/s)", "Max (TB/s)", "Mean (TB/s)"]
        table1.add_row(copy)
        table1.add_row(mul)
        table1.add_row(add)
        table1.add_row(triad)
        table1.add_row(dot)
        print(table1)
        tools.export_markdown("HBM Bandwidth", "HBM bandwidth Results", table1)

        tools.post_benchmark_entry(tools.create_bm_entry("Copy", self.name, self.machine_name, copy[-1]))
        tools.post_benchmark_entry(tools.create_bm_entry("Mul", self.name, self.machine_name, mul[-1]))
        tools.post_benchmark_entry(tools.create_bm_entry("Add", self.name, self.machine_name, add[-1]))
        tools.post_benchmark_entry(tools.create_bm_entry("Triad", self.name, self.machine_name, triad[-1]))
        tools.post_benchmark_entry(tools.create_bm_entry("Dot", self.name, self.machine_name, dot[-1]))
