import json
import os
import shlex
import subprocess
import datetime
import time
import csv 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from Infra import tools
from prettytable import PrettyTable


class GEMMCublastLt:
    def __init__(self, path: str, machine: str, b: int = 1, i: int = 1000, w: int = 10000):
        self.name = "GEMMCublasLt"
        config = self.get_config(path)
        self.m, self.n, self.k, self.duration, self.datatype = self.config_conversion(config)
        self.b = b
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
        self.buffer = []

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def parse_json(self, config, var):
        if var == "duration":
            return config["inputs"]["duration"]
        if var == "datatype":
            return config["inputs"]["datatype"]
        start = config["inputs"][var]["start"]
        end = config["inputs"][var]["end"]
        interval = config["inputs"][var]["interval"]
        data = [a for a in range(start, end, interval)]
        if not data or data[-1] < end:
            data.append(end)
        return data

    def config_conversion(self, config):
        m = self.parse_json(config, "m")
        n = self.parse_json(config, "n")
        k = self.parse_json(config, "k")
        duration = self.parse_json(config, "duration")
        datatype = self.parse_json(config, "datatype")

        return m, n, k, duration, datatype

    def build(self):
        bindir = tools.create_dir("bin")
        self.bindir = bindir

        path = "superbenchmark"
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/gitaumark/superbenchmark",
                    path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(results.stderr.decode('utf-8'))

        current = os.getcwd()

        build_path = os.path.join(
            current,
            "superbenchmark/superbench/benchmarks/micro_benchmarks/cublaslt_gemm",
        )
        os.chdir(build_path)

        results = subprocess.run(
            ["cmake", "-S", "./"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(results.stderr.decode('utf-8'))
        results = subprocess.run(
            ["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(results.stderr.decode('utf-8'))
        results = subprocess.run(
            ["mv", "cublaslt_gemm", bindir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.chdir(current)

    # if no shmoo data found, run shmoo. if found, use existing shmoo data
    def run_shmoo(self):
        if os.path.isfile('Outputs/GEMMCublasLt_Shmoo_'+ self.machine_name +'_' +self.datatype+'.csv'):
            self.buffer = self.parse_csv('Outputs/GEMMCublasLt_Shmoo_'+ self.machine_name +'_' +self.datatype+'.csv')
        else:
            self.buffer = self.run() 

        self.plot_shmoo()   

    def run_nvml(self):
        # run nvidia-smi -q -d SUPPORTED_CLOCKS to establish the clock frequency to use

        # res = subprocess.run(["sudo", "nvidia-smi", "-ac", "3201,1980"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(res.stdout.decode('utf-8'))
        cmd = (
            f"nvidia-smi --query-gpu=timestamp,utilization.gpu,"
            + f"power.draw,enforced.power.limit,clocks.current.sm,"
            + f"clocks.current.memory,temperature.gpu --format=csv --id=0 -lms 100 -f Outputs/{self.name}_power.csv"
        )


        cmd = shlex.split(cmd)
    
        power_results = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        gemm_data = self.run_gemm()

        cmd = "pkill -9 nvidia-smi"
        cmd = shlex.split(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        power_results.communicate()

        self.plot_power_data(gemm_data, self.parse_power_data("GEMMCublasLt_power.csv"))


    # run GEMM Sweep, start and end dims can be altered in config.json
    # results saved in Outputs folder
    def run(self) -> list:
        print("Running GEMM Sweep...")
        current = os.getcwd()
        os.chdir(self.bindir)

        end_interval = str(self.m[-1])

        tot_time = 0
        buffer = []
        with open('../Outputs/GEMMCublasLt_Shmoo_'+ self.machine_name +'_' +self.datatype+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["M", "N", "K", "Batch", "Time(us)", "TFLOPS"])
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for i in range(len(self.m)):
                for j in range(len(self.n)):
                    for t in range(len(self.k)):
                        a = str(self.m[i]) == end_interval
                        b = str(self.n[j]) == end_interval 
                        c = str(self.k[t]) == end_interval

                        if (a and b) or (b and c) or (a and c):
                            start.record()
                            results = subprocess.run(
                                [
                                    "./cublaslt_gemm",
                                    "-m",
                                    str(self.m[i]),
                                    "-n",
                                    str(self.n[j]),
                                    "-k",
                                    str(self.k[t]),
                                    "-b",
                                    str(self.b),
                                    "-i",
                                    str(self.i),
                                    "-w",
                                    str(self.w),
                                    "-t",
                                    self.datatype,
                                    "-r",
                                    str(1),
                                    "-s",
                                    str(-1),
                                    "-e",
                                    str(1),
                                ],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )

                            end.record()
                            torch.cuda.synchronize()
                            log = results.stdout.decode("utf-8")
                            # print("m: ", self.m[i] ," n: ", self.n[j], " k: ", self.k[t])
                            # handle errors and failed cases
                            if log.find("failed") != -1 or log.find("error") != -1:
                                continue
                            buffer.append(log.split())
                            writer.writerow(log.split())
                            tot_time = tot_time + start.elapsed_time(end)
        os.chdir(current)
        return buffer
    

    def run_gemm(self):
            current = os.getcwd()
            os.chdir(self.bindir)
            
            t_end = time.time() + self.duration
            buffer = []
            while time.time() < t_end:
                results = subprocess.run(
                    [
                        "./cublaslt_gemm",
                        "-m",
                        str(8192),
                        "-n",
                        str(8192),
                        "-k",
                        str(8192),
                        "-b",
                        str(self.b),
                        "-i",
                        str(self.i),
                        "-w",
                        str(self.w),
                        "-t",
                        self.datatype,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                log = results.stdout.decode('utf-8').split()
                curr_time = self.parse_timestamp(str(datetime.datetime.now()).split(' ')[1])
                log.append(curr_time)
                buffer.append(log)
            os.chdir(current)
            return buffer
    

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        print("Running CublasLt...")
        current = os.getcwd()
    
        m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816, 802816] 
        n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 192, 768]

        os.chdir(self.bindir)
        
        buffer = []
        
        
        for i in range(len(m_dims)):
            results = subprocess.run(
                [
                    "./cublaslt_gemm",
                    "-m",
                    str(m_dims[i]),
                    "-n",
                    str(n_dims[i]),
                    "-k",
                    str(k_dims[i]),
                    "-b",
                    str(self.b),
                    "-i",
                    str(self.i),
                    "-w",
                    str(self.w),
                    "-t",
                    self.datatype,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log = results.stdout.decode('utf-8').split()
            buffer.append(log)

        table1 = PrettyTable()  

        with open('../Outputs/GEMMCublasLt_Performance_' + self.machine_name + '_' + self.datatype+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["M", "N", "K", "Batch", "Time(us)", "TFLOPS"])
            table1.field_names = ["M", "N", "K", "Batch Size", "Time(us)", "TFLOPS"]
            for item in buffer:
                writer.writerow(item)
                table1.add_row(item)

        print(table1)
        os.chdir(current)


    def parse_power_data(self, filename):
        power_data  = []
        current = os.getcwd()
        os.chdir(os.path.join(
            current,
            "Outputs",
        ))
        with open(filename, mode='r') as csvfile:
            line_num = 0
            for line in csvfile:
                
                if line_num > 0:
                    temp_line = line.split(',')
                    res = []
                    if len(temp_line) < 7:
                        continue
                    res.append(self.parse_timestamp(temp_line[0].split()[1]))
                    res.append(float(temp_line[2].split()[0]))
                    res.append(float(temp_line[4].split()[0]))
                    res.append(float(temp_line[3].split()[0]))
                    res.append(float(temp_line[6].split()[0]))
                    power_data.append(res)
                line_num += 1
        os.chdir(current)
        return power_data


    def parse_timestamp(self, timestamp):
        times = timestamp.split(":")
        total_time = 0
        total_time += int(times[0]) * 60 * 60
        total_time += int(times[1]) * 60
        total_time += float(times[2])
        return total_time

    def plot_power_data(self, gemm_data, power_data):
        power_limit = power_data[0][3]

        baseline = float(power_data[0][0])
        t1 = np.array(power_data)[:, 0].astype(float)
        p = np.array(power_data)[:, 1].astype(float)
        freq = np.array(power_data)[:, 2].astype(int)
        temp = np.array(power_data)[:, 4].astype(int)
        t2 = np.array(gemm_data)[:, -1].astype(float)
        g = np.array(gemm_data)[:, -2].astype(float)
        t1 = t1 - baseline
        t2 = t2 - baseline

        fig, (ax, ax2, ax3, ax4)= plt.subplots(4, figsize=(18,18))
     

        ax.plot(t1, p, c="red")
        ax.grid(True)
        ax.set_ylim([50, power_limit + 100])
        ax.axhline(y=power_limit, color='r', linestyle='--', label='Power Limit')
        ax.set_ylabel("Power Draw (W)",fontsize=14)
        ax.legend(loc="upper right")

        ax2.plot(t1, freq)
        ax2.grid(True)
        ax2.set_ylim([300, 2000])
        ax2.set_ylabel("Clock Frequency (MHz)", fontsize=14)

        ax3.plot(t1, temp, c="red")
        ax3.grid(True)
        ax3.set_ylim([20, 100])
        ax3.set_ylabel("Temperature (C)", fontsize=14)
        ax3.set_xlabel("Time (seconds)")

        ax4.boxplot(g)
        ax4.set_xticks([])
        ax4.set_ylim([1000, 1300])
        ax4.set_ylabel("Performance (TFLOPS)", fontsize=14)
        ax4.grid(True)
        
        fig.suptitle("8K GEMM Measurements " + self.machine_name, fontsize=28)
        plt.savefig("Outputs/GEMMCublasLt_Power_"+self.machine_name + "_" +self.datatype+".png", format="png", bbox_inches="tight")
        plt.close()

 
    # // m n k batch time_us tflops

    def parse_csv(self, filename):
        buffer = []
        with open(filename, mode="r") as csvFile:
            line_num = 0
            for line in csvFile:
                if line_num > 0:
                    buffer.append(line.strip().split(','))
                line_num += 1
        return buffer
                

    def plot_shmoo(self):
        arr = np.array(self.buffer)


        # splitting up the data into m, n, k sweeps
        m_arr = []
        n_arr = []
        k_arr = []

        
        # size of the other 2 dims that are constant
        dim_size = '4096'
        for i in range(len(arr)):
            if arr[i][1] == dim_size and arr[i][2] == dim_size:
                m_arr.append(arr[i])
            if arr[i][0] == dim_size and arr[i][2] == dim_size:
                n_arr.append(arr[i])
            if arr[i][0] == dim_size and arr[i][1] == dim_size:
                k_arr.append(arr[i])

        
        m_arr = np.array(m_arr)
        n_arr = np.array(n_arr)
        k_arr = np.array(k_arr)

        # plot M Shmoo
        x = m_arr[:, 0].astype(int)
        y = m_arr[:, 5].astype(float)

       
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.grid(True)
        ax.set_title("4096, 4096 NT GEMM M Shmoo")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        plt.xlabel("M Dim")
        plt.ylabel("TFLOPS")
        plt.savefig("Outputs/GEMMCublasLt M Shmoo_" + self.machine_name + "_" + self.datatype + ".png", bbox_inches="tight")
        plt.close(fig)
        
        # plot N Shmoo
        x = n_arr[:, 1].astype(int)
        y = n_arr[:, 5].astype(float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.grid(True)
        ax.set_title("4096, 4096 NT GEMM N Shmoo")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        plt.xlabel("N Dim")
        plt.ylabel("TFLOPS")
        plt.savefig("Outputs/GEMMCublasLt N Shmoo_" + self.machine_name + "_" + self.datatype + ".png", bbox_inches="tight")
        plt.close(fig)

        # plot K shmoo
        x = k_arr[:, 2].astype(int)
        y = k_arr[:, 5].astype(float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.grid(True)
        ax.set_title("4096, 4096 NT GEMM K Shmoo")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        plt.xlabel("K Dim")
        plt.ylabel("TFLOPS")
        plt.savefig("Outputs/GEMMCublasLt K Shmoo_" + self.machine_name + "_" + self.datatype + ".png", bbox_inches="tight")
        plt.close()
