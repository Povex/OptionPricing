import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

CPU = df.query("Engine == 'SerialCPU' & Type == 'EuropeanCall' ")
GPU = df.query("Engine == 'GPU' & Type == 'EuropeanCall' ")

timeElapsedCPU = CPU["timeElapsed[s]"]
timeElapsedGPU = GPU["timeElapsed[s]"]
nSimulations = CPU["nSimulations"]

plt.plot(nSimulations, timeElapsedCPU, label = "Serial CPU", color='blue')
plt.plot(nSimulations, timeElapsedGPU, label = "Parallel GPU", color='orange')

plt.grid()
plt.legend()
plt.title("European option - Time elapsed", fontsize=17)
plt.ylabel("Time elapsed [s]", fontsize=14)
plt.xlabel("Number of simulations", fontsize=14)

plt.savefig('plot.png')

plt.clf()
speedup = [x/y for x,y in zip(timeElapsedCPU, timeElapsedGPU)]
plt.plot(nSimulations, speedup, label = "CPU/GPU time elapsed", color='blue')
plt.grid()
plt.legend()
plt.title("European option - Speedup", fontsize=17)
plt.ylabel("Speedup", fontsize=14)
plt.xlabel("Number of simulations", fontsize=14)

plt.savefig('plot-speedup.png')