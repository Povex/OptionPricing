import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

CPU = df.query("Engine == 'SerialCPU' ")
GPU = df.query("Engine == 'GPU' ")

nBinaryOptions = np.arange(1, 365, 1)

timeElapsedCPU = CPU["timeElapsed"]
timeElapsedGPU = GPU["timeElapsed"]

plt.plot(nBinaryOptions[:127], timeElapsedCPU, label = "Serial CPU")
plt.plot(nBinaryOptions, timeElapsedGPU, label = "GPU")

plt.grid()
plt.legend()
plt.ylabel("Time elapsed [s]")
plt.xlabel("N binary options")

plt.savefig('plot.png')