import json
import matplotlib.pyplot as plt


def plotdata(gput:list, cput:list, gpuacc:list, cpuacc:list) -> None:
    plt.plot(gput, gpuacc, 'x--',  label='GPU')
    plt.plot(cput, cpuacc, 'o--', label='CPU')
    plt.title("accuracy over executiontime for GPU and CPU")
    plt.ylabel("accuracy in %")
    plt.xlabel("time in s")
    plt.legend()
    plt.savefig("timeplot_3_1.png")

with open("GPUdata.json", 'r') as json_file:
    GPUdata = json.load( json_file)

with open("CPUdata.json", 'r') as json_file:
    CPUdata = json.load( json_file)


plotdata(gput=GPUdata['time'], cput=CPUdata['time'], gpuacc=GPUdata['acc'], cpuacc=CPUdata['acc'])
