import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

serialCPU = df.query("Engine == 'SerialCPU' & Type == 'EuropeanCall' ")
GPU = df.query("Engine == 'GPU' & Type == 'EuropeanCall' ")
CPU = df.query("Engine == 'CPU' & Type == 'EuropeanCall' ")

timeElapsedserialCPU = serialCPU["timeElapsed[s]"]
timeElapsedGPU = GPU["timeElapsed[s]"]
timeElapsedCPU = CPU["timeElapsed[s]"]
nSimulations = CPU["nSimulations"]

plt.plot(nSimulations, timeElapsedserialCPU, label = "Serial CPU", color='blue')
plt.plot(nSimulations, timeElapsedGPU, label = "Parallel GPU", color='orange')
plt.plot(nSimulations, timeElapsedCPU, label = "Parallel CPU", color='green')

plt.grid()
plt.legend()
plt.title("European option - Time elapsed", fontsize=17)
plt.ylabel("Time elapsed [s]", fontsize=14)
plt.xlabel("Number of simulations", fontsize=14)

plt.savefig('plot.png')

plt.clf()
speedupGPU = [x/y for x,y in zip(timeElapsedserialCPU, timeElapsedGPU)]
speedupCPU = [x/y for x,y in zip(timeElapsedserialCPU, timeElapsedCPU)]
plt.plot(nSimulations, speedupGPU, label = "serial(CPU)/parallel(GPU) time", color='orange')
plt.plot(nSimulations, speedupCPU, label = "serial(CPU)/parallel(CPU) time", color='green')
plt.grid()
plt.legend()
plt.title("European option - Speedup", fontsize=17)
plt.ylabel("Speedup", fontsize=14)
plt.xlabel("Number of simulations", fontsize=14)

plt.savefig('plot-speedup.png')