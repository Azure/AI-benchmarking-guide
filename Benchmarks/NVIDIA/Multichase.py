import json
import os
import statistics
import subprocess
import time
import csv
from Infra import tools
from prettytable import PrettyTable

class Multichase:
    def __init__(self, path:str, machine: str):
        self.name = "Multichase"
        self.machine_name = machine
    
    def build(self):
        current = os.getcwd()
        path = "multichase"
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_command(['git', 'clone', 'https://github.com/google/multichase', path])

            build_path = os.path.join(current, "multichase")
            os.chdir(build_path)
            
            results = tools.run_command('make', shell=True)
            os.chdir(current)

    def run(self):
        current = os.getcwd()
        tools.progress("Running Multichase...")

        results = tools.run_command(["bash", "run_multichase.sh"], cwd="Benchmarks/NVIDIA", capture_output=True)
        if not results.stdout.strip():
            raise tools.runtime.OutputError("Multichase returned no measurements")
        tools.detail(results.stdout.decode("utf-8"))
        tools.export_markdown("Multichase", results.stdout.decode("utf-8"), None)
