import json
import subprocess
import os
import torch
import time
import statistics

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#from matplotlib.ticker import 

from Infra import sqlite


import numpy as np

class NVBandwidth:
    def __init__(self, path:str, machine: str):
        
        self.name='NVBandwidth'
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
        return config['inputs']['num_runs'], config['inputs']['interval']
    

    def config_conversion(self, config)->tuple[list, list, list]:
        return self.parse_json(config)
        
    def build(self):
        current = os.getcwd()
        
        path ='nvbandwidth'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nvbandwidth', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        build_path = os.path.join(current, 'nvbandwidth')
        os.chdir(build_path)
        results = subprocess.run(['sed', '-i', '2i\set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)', 'CMakeLists.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results = subprocess.run(['sudo', './debian_install.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(results.stderr.decode('utf-8'))          
        os.chdir(current)
        


    def run(self): 
        current = os.getcwd()
        os.chdir(os.path.join(current, 'nvbandwidth'))
        print("Running NVBandwidth...")
 
        buffer=[]
            
        results = subprocess.run('./nvbandwidth -t device_to_device_bidirectional_memcpy_read_sm | grep -A 11 "Running device_to_device_bidirectional_memcpy_read_sm."', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = results.stdout.decode('utf-8')
        print(results.stderr.decode('utf-8')) 
        buffer.append(log)
        
        results = subprocess.run('./nvbandwidth -t device_to_host_memcpy_sm | grep -A 4 "Running device_to_host_memcpy_sm."', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = results.stdout.decode('utf-8')
        print(results.stderr.decode('utf-8')) 
        buffer.append(log)
       
        results = subprocess.run('./nvbandwidth -t host_to_device_memcpy_sm | grep -A 4 "Running host_to_device_memcpy_sm."', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = results.stdout.decode('utf-8')
        print(results.stderr.decode('utf-8')) 
        buffer.append(log)
      
        os.chdir(current)
    
        file = open("Outputs/NVBandwidth_" + self.machine_name + ".txt", "w")
        for item in buffer:
            file.write(item)
            print(item)
     
        
        self.buffer=buffer
        os.chdir(current)
