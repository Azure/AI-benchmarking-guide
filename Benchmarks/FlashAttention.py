import subprocess
import os

class FlashAttention:
    def __init__(self, path:str, machine: str):
        
        self.name='FlashAttention'
        self.machine_name = machine

        self.buffer = []
    
       
    def run(self):
        current = os.getcwd()
        
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run('git clone https://github.com/Dao-AILab/flash-attention.git',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        build_path = os.path.join(current, 'flash-attention/benchmarks')
        os.chdir(build_path)
        print("Running Flash Attention...")
        results = subprocess.run('python3 benchmark_flash_attention.py | grep -A 2 "### causal=False, headdim=128, batch_size=2, seqlen=8192 ###"',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.chdir(current)
        file = open("Outputs/FlashAttention_" + self.machine_name + ".txt", "w")
        res = results.stdout.decode('utf-8').split("\n")
        print(res[1])
        print(res[2])
        file.write(res[1] + "\n")
        file.write(res[2]) 
