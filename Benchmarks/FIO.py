import os
import subprocess

class FIO:
    def __init__(self, path: str, machine: str):
        self.name = "FIO"
        self.machine_name = machine
        
    def run(self):
        current = os.getcwd()
        
        print("Running FIO Tests...")
        tests = [
            ["read", "1M"],
            ["read", "512k"],
            ["read", "1k"],
            ["write", "1M"],
            ["write", "512k"],
            ["write", "1k"],
            ["randwrite", "1k"],
            ["randread", "1k"]
        ]
        file = open('Outputs/FIO_results_' + self.machine_name +'.txt', 'w')

        for test in tests:
            results = subprocess.run(
                "fio --bs=" + test[1] +  " --ioengine=libaio --iodepth=255 --directory=" + current + "/Outputs --direct=1 --runtime=300 --numjobs=4 --rw=" +test[0]+ " --name=test --group_reporting --gtod_reduce=1 --size=10G | grep -A 1 ': bw='",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            res = results.stdout.decode('utf-8').split()[2].strip(",()")
            res = test[0] + " BS=" + test[1] + ": " + res
            print(res)
            file.write(res + '\n')
        
        file.close()   
        results = subprocess.run(
            "rm Outputs/test*",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
