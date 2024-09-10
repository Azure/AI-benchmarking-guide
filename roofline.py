import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import pandas as pd
import matplotlib.patches as mpatches
from random import randrange


# parameters
font = { 'size'   : 15}
plt.rc('font', **font)
markersize = 10 #12

# markers
colors = ['b','r','g','m','y','c']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

# mycolor
pytorch    = '#136aa8'
tensorflow = '#ff7527'
mycolors   = [pytorch, tensorflow]

# styledict
styledict = {"thorsten": {"fontsize_annotation": 10, "roof_color": 'gray', "legend_points_ncol": 2, "frameon": False}, 
             "charlene": {"fontsize_annotation": 13, "roof_color": 'k', "legend_points_ncol": 1, "frameon": True}}

#plot H100 roofs
def plot_h100_roofs(fig, ax, xlim, ylim, styledict, datatype, scaling = 1.):
    #extract general settings
    fontsize_annotation = styledict["fontsize_annotation"] #10
    roof_color = styledict["roof_color"]


    #set up theoretical roofs
    #mem
    smemroofs = [3350.00]
    smem_roof_name = ['HBM Theo']
    
    #flops
    if datatype == 'fp8e4m3':
        scomproofs_fp16 = [3958]
        scomp_roof_name_fp16 = ['FP8 Theo']
    else:
        scomproofs_fp16 = [1979]
        scomp_roof_name_fp16 = ['FP16 Theo']
    scalingFactorForRoofs = scaling
    
    #resolution
    #nx = 10000
    xmin = xlim[0]
    xmax = xlim[1]
    ymin = ylim[0]
    ymax = ylim[1]
    nx = 10*xmax
    
    #set limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    #plot roofs:
    dx = (xmax-xmin)/nx
    for idm, smem in enumerate(smemroofs):
        #fp16
        for idc, scomp in enumerate(scomproofs_fp16):
            xvals = np.arange(xmin, xmax, dx)
            yvals = np.minimum(smem*xvals*10**-3, np.tile(np.array(scomp),len(xvals)))
            ax.plot(xvals, yvals, c=roof_color, ls='-', lw='2')
            if idm==0:
                #plot scomp
                label = scomp_roof_name_fp16[idc] + ': ' + '{0:.1f}'.format(float(scomp)) + ' TFLOP/s'
                ax.annotate(label, xy=(3200,scomp), xytext=(-5,5), textcoords="offset points", color=roof_color, horizontalalignment='right', fontsize=fontsize_annotation)
        
        #plot mem
        #find intersection
        scomp = scomproofs_fp16[0]
        yis = ymin
        xis = minimize(fun = lambda x: np.abs(min([smem*x*10**-3,scomp])-yis), x0=xmin, tol=1.e-10)["x"][0]
        #find elbow
        optimize = minimize(fun=lambda x: np.abs(smem*x*10**-3-scomp), x0=xmin, tol=1.e-10)
        xelb = optimize["x"][0]
        yelb = min(smem*xelb*10**-3, scomp)
        #angle in plot coord system
        ang = np.rad2deg( np.arctan2(yelb-yis, xelb-xis) )
        print("angle ", ang)
        #angle in figure coord system
        pts = np.array((xelb, yelb)).reshape((1,2))
        trans_ang = ax.transData.transform_angles(np.array((ang,)), pts)[0]
        #ax.plot(xis, yis, marker="o", ms=10)
        #ax.plot(xelb, yelb, marker="o", ms=10)
        label = smem_roof_name[idm] + ': ' + '{0:.1f}'.format(float(smem)/scalingFactorForRoofs) + ' GB/s'
        if datatype == "fp8e4m3":
            xy = (470, 2800)
        else:
            xy = (200, 800)
        # ax.annotate(label, xy=xy, xytext=(5,5), color=roof_color, \
        #         rotation=trans_ang, rotation_mode='anchor', \
        #         horizontalalignment='right', \
        #         verticalalignment='bottom', \
        #         textcoords="offset points", \
        #         fontsize=fontsize_annotation)
        
        T0    = 4.7e-6;   # kernel overhead estimate
        Rpeak = smemroofs[0] * 1e9; # streaming DEVICE bw
        if datatype == 'fp8e4m3':
            Fpeak = scomproofs_fp16[0] * 1.008e12;  # high percentage of peak DEVICE GFLOPS/s
        else:
            Fpeak = scomproofs_fp16[0] * 1.009e12;
        
        # grid of AI and B values
        maxAI = 400.1*Fpeak/Rpeak;    # maximum arithmetic intensity (AI)
        maxB = 1e13;    # maximum number of bytes
        
        NAI = 5000;     # number of AI samples
        NB = 200;      # number of bytes values to sample
        
        _AI = np.linspace(0,maxAI,NAI) #'*ones(1,NB);
        _B = 10**np.linspace(2, 9, NB);
        
        AI, B = np.meshgrid(_AI, _B)
        
        # model for execution time
        
        cost_per_byte = np.maximum(1/Rpeak, AI/Fpeak)
        #flops/bype/(flops/s) = s/byte
        TB = T0 + B*cost_per_byte;
        
        # model for throughput in GLOPS
        GFLOPS = (B*AI/TB)/1e12;
        
        cmap_reversed = matplotlib.colormaps['tab20c_r']
        
        #fig = plt.figure()
        #ax = fig.gca()
        im = ax.pcolormesh(AI, GFLOPS, np.log10(B), cmap=cmap_reversed, shading="auto")       
        fig.colorbar(im, ax=ax, label=r"$\log_{10}$(bytes)")

def arithmetic_intensity(m,n,k):
    flops = 2*m*n*k
    byte_accesses = 2*((m*k)+(n*k)+(m*n))
    return flops/byte_accesses

def plot_points(data, color, title, datatype):
    #pick style:
    style = styledict["charlene"]

    #figure stuff
    fig = plt.figure(1,figsize=(12.67,6.6))
    plt.clf()
    ax = fig.gca()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Arithmetic Intensity [FLOP/Byte]')
    ax.set_ylabel('Performance [TFLOP/s]')
    xmin = -0.1 #np.floor(np.log(min(AI_l1))) #-2
    xmax = 4.5 #np.ceil(np.log(max(AI_dram)))
    ymin = 1 #10./scalingFactorForRoofs #10.0 / scalingFactorForRoofs
    ymax = 2000.0*5.5 #max(FLOPs)*2./scalingFactorForRoofs

    #some handles
    marker_handles = []

    if datatype == "fp8e4m3":
        y_lim = 4200
    else:
        y_lim = 2100

    #plot roofs
    plot_h100_roofs(fig, ax, (0, 6000), (0, y_lim), style, datatype=datatype)

    #H100 data
    cublaslt_results = []
    for i in range(len(data)):
        temp_data = []
        for j in range(len(data[i])):
            temp_data.append(float(data[i][j]))
        #print(temp_data)
        cublaslt_results.append(temp_data)
        m = temp_data[0]
        n = temp_data[1]
        k = temp_data[2]
        cublaslt_results[i].append(arithmetic_intensity(m,n,k))
        print(cublaslt_results[i][6], ", ",cublaslt_results[i][5])




    h100 = cublaslt_results
    

    #plot data
    #generate style with color and marker for each Type
    style = {}
    style["color"] = {"BERT" : 'red', "OPT-175B" : 'blue', "OPT-1/2T" : 'red'}
    style["marker"] = {"FP16" : 'o', "FP8" : 'x'}
    #add points to ax plot with ai on x-axis and tflops on y-axis, color by model and shape by datatype from h100 dataframe
    colors = ["b", "g", "r", "c", "lime", "darkviolet"]
    
    if title == "Model_sizes":
        for i in range(0,len(h100)):
            #marker_handles.append(ax.plot(h100[i][7],h100[i][6],c=colors[i], marker=".", markersize=12, label="m="+str(h100[i][1])+", n="+str(h100[i][2])+", k="+str(h100[i][3])))
            marker_handles.append(ax.plot(h100[i][6],h100[i][5],c=colors[i % len(colors)], marker='.', markersize=15))
    
        #add legend for color and shape to plot

        legend = ['roofline']
        for item in h100:
            res = "m="
            res += str(int(item[0]))
            res += " n="
            res += str(int(item[1]))
            res += " k="
            res += str(int(item[2]))
            legend.append(res)
        
        ax.legend(legend, loc='lower right', fontsize=7)

    else:
        for i in range(0,len(h100)):
            #marker_handles.append(ax.plot(h100[i][7],h100[i][6],c=colors[i], marker=".", markersize=12, label="m="+str(h100[i][1])+", n="+str(h100[i][2])+", k="+str(h100[i][3])))
            marker_handles.append(ax.plot(h100[i][6],h100[i][5],c=color, marker='.', markersize=12))       

    plt.tight_layout()
    plt.savefig('Outputs/' + title + "_roofline_" +datatype+".png")
    #plt.close()
