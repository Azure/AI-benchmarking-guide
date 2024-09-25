import json
import os
import csv 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import subprocess
import time
import csv
import numpy as np
import torch
from prettytable import PrettyTable


class NCCLBandwidth:
    def __init__(self, path:str, machine: str):
        
        self.name='NCCLBandwidth'
        self.machine_name = machine
        config = self.get_config(path)
        self.start, self.end, self.num_gpus = self.config_conversion(config)
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
        return config['inputs']['start'], config['inputs']['end'], config['inputs']['num_gpus']
    

    def config_conversion(self, config)->tuple[list, list, list]:
        return self.parse_json(config)
        
    def build(self):
        current = os.getcwd()
        
        path ='nccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nccl-tests.git', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl-tests')
            os.chdir(build_path)
            results = subprocess.run(['make'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(results.stderr.decode('utf-8')) 
        else:
            build_path = os.path.join(current, 'nccl-tests')
            os.chdir(build_path)
      


    def run(self): 
        current = os.getcwd()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        runs = ["Tree", "Ring", "NVLS", "NVLSTree"]

        print("Running NCCL AllReduce...")
        for run in runs:
            start.record()
            results = subprocess.run('NCCL_ALGO='+ run +' ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 40 | grep float', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(results.stderr.decode('utf-8'))
            end.record()
            torch.cuda.synchronize()
            res = results.stdout.decode('utf-8').split('\n')
            log = []
            for line in res:
                line = line.split()
                if len(line) == 13:
                    log.append(line[11])
            
            buffer.append(log)
        
        table1 = PrettyTable()
        runs = ["Message Size", "Tree", "Ring", "NVLS", "NVLSTree"]

        for i in range(len(buffer)):
            table1.add_column(runs[i], buffer[i])
        
        print(table1)
        
        
        self.buffer=buffer
        self.save()
        os.chdir(current)

    def plot(self):
        x = np.arange(len(self.buffer[0][12:]))
        y1 = np.array(self.buffer[1]).astype(float)[12:]
        y2 = np.array(self.buffer[2]).astype(float)[12:]
        y3 = np.array(self.buffer[3]).astype(float)[12:]
        y4 = np.array(self.buffer[4]).astype(float)[12:]
        width = 0.8
        widt = 0.2
        
        plt.bar(x-width, y1, widt) 
        plt.bar(x, y2, widt) 
        plt.bar(x+width, y3, widt) 
        plt.bar(x+(2*width), y4, widt)
        plt.bar(x+(3*width), 0, widt) 
        plt.savefig("../Outputs/NCCLres.png")
        self.save()

    def save(self):
        with open('../Outputs/NCCLBandwidth_' + self.machine_name + '.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            for i in range(len(self.buffer[0])):
                row = [self.buffer[0][i], self.buffer[1][i], self.buffer[2][i], self.buffer[3][i], self.buffer[4][i]]  
                writer.writerow(row)
    
