import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

inputFile = "data.csv"
df = pd.read_csv(inputFile)

barrierOffset = df["barriersOffset"]
value = df["value"]

plt.plot(barrierOffset, value, label = "Actualized payoff")

plt.grid()
plt.legend()
plt.ylabel("Actualized payoff")
plt.xlabel("Offset")

plt.savefig('plot.png')