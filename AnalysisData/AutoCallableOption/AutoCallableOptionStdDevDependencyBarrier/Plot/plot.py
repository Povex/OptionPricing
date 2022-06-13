import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

constantBarrier = df["constantBarrier"]
value = df["stdError"]

plt.plot(constantBarrier, value, label = "Standard error")

plt.grid()
plt.legend()
plt.ylabel("Standard error")
plt.xlabel("Constant barrier")

plt.savefig('plot.png')