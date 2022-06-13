import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

constantBarrier = df["constantBarrier"]
value = df["value"]

plt.plot(constantBarrier, value, label = "Actualized payoff")

plt.grid()
plt.legend()
plt.ylabel("Actualized payoff")
plt.xlabel("Constant barrier")

plt.savefig('plot.png')