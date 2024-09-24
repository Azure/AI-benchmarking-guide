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
import roofline
from Infra import tools


def parse_csv(filename):
    buffer = []
    with open(filename, mode="r") as csvFile:
        line_num = 0
        for line in csvFile:
            if line_num > 0:
                buffer.append(line.strip().split(','))
            line_num += 1
    return buffer


# plot two shmoos next to each other, from 2 csvs
def plot_shmoo(filename, filename2):
    arr = np.array(parse_csv(filename))
    arr2 = np.array(parse_csv(filename2))
    #arr = np.array(self.parse_csv("Outputs/GEMMCublasLt_Performance.csv"))

    # splitting up the data into m, n, k sweeps
    m_arr = []
    n_arr = []
    k_arr = []

    m_arr2 = []
    n_arr2 = []
    k_arr2 = []

    
    # size of the other 2 dims that are constant
    dim_size = '4096'
    for i in range(len(arr)):
        if arr[i][1] == dim_size and arr[i][2] == dim_size:
            m_arr.append(arr[i])
        if arr[i][0] == dim_size and arr[i][2] == dim_size:
            n_arr.append(arr[i])
        if arr[i][0] == dim_size and arr[i][1] == dim_size:
            k_arr.append(arr[i])


    for i in range(len(arr2)):
        if arr2[i][1] == dim_size and arr2[i][2] == dim_size:
            m_arr2.append(arr2[i])
        if arr2[i][0] == dim_size and arr2[i][2] == dim_size:
            n_arr2.append(arr2[i])
        if arr2[i][0] == dim_size and arr2[i][1] == dim_size:
            k_arr2.append(arr2[i])
    
    m_arr = np.array(m_arr)
    n_arr = np.array(n_arr)
    k_arr = np.array(k_arr)

    m_arr2 = np.array(m_arr2)
    n_arr2 = np.array(n_arr2)
    k_arr2 = np.array(k_arr2)
    

    # plot M Shmoo
    print("M ARR")
    x = m_arr[:, 0].astype(int)
    y = m_arr[:, 5].astype(float)

    x2 = m_arr2[:, 0].astype(int)
    y2 = m_arr2[:, 5].astype(float)

    
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, y, color="red", label="ND H200 v5")
    ax.plot(x2, y2, color="blue", label="ND H100 v5")
    ax.legend()
    ax.grid(True)
    fig.suptitle("4K GEMM M Shmoo", fontsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
   
    ax.set_xlabel("M Dim")
    ax.set_ylabel("TFLOPS")
    plt.savefig("Outputs/GEMMCublasLt M Shmoo_comparison.png", bbox_inches="tight")
    plt.close(fig)
    
    # plot N Shmoo
    print("N ARR")
    x = n_arr[:, 1].astype(int)
    y = n_arr[:, 5].astype(float)

    x2 = n_arr2[:, 1].astype(int)
    y2 = n_arr2[:, 5].astype(float)

    
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, y, color="red", label="ND H200 v5")
    ax.plot(x2, y2, color="blue", label="ND H100 v5")
    ax.legend()
    ax.grid(True)
    fig.suptitle("4K GEMM N Shmoo", fontsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    
    ax.set_xlabel("N Dim")
    ax.set_ylabel("TFLOPS")
    plt.savefig("Outputs/GEMMCublasLt N Shmoo_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # plot K shmoo
    print("K ARR", arr)
    x = k_arr[:, 2].astype(int)
    y = k_arr[:, 5].astype(float)

    x2 = k_arr2[:, 2].astype(int)
    y2 = k_arr2[:, 5].astype(float)

    
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, y, color="red", label="ND H200 v5")
    ax.plot(x2, y2, color="blue", label="ND H100 v5")
    ax.legend()
    ax.grid(True)
    fig.suptitle("4K GEMM K Shmoo", fontsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.set_xlabel("K Dim")
    ax.set_ylabel("TFLOPS")
    plt.savefig("Outputs/GEMMCublasLt K Shmoo_comparison.png", bbox_inches="tight")
    plt.close(fig)

# replace with the paths of H200 and H100 shmoo file paths
plot_shmoo("Outputs/GEMMCublasLt_Performance_NVIDIA H200_fp8e4m3.csv", "Outputs/GEMMCublasLt_ShmooPerformance_NVIDIA H100 80GB HBM3_fp8e4m3.csv")
