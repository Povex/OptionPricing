import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

nSimulations = 1048576
nObsDates = np.arange(1, 366, 1)

stdError = df["stdError"]
confidence1 = df["confidence1"]
confidence2 = df["confidence2"]

plt.plot(nObsDates, stdError, label = "Standard error")

plt.grid()
plt.legend()
plt.ylabel("Standard Error")
plt.xlabel("N binary options")

plt.savefig('plot.png')