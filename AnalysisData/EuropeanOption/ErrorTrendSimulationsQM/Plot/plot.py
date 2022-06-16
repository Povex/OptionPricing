import numpy as np
from matplotlib import pyplot as plt
import os.path
import pandas as pd

import math
# Call payoff price and error

inputFile = "data.csv"
df = pd.read_csv(inputFile)

CPU = df.query("Engine == 'SerialCPU' & Type == 'EuropeanCall' ")
GPU = df.query("Engine == 'GPU' & Type == 'EuropeanCall' ")
valueCPU = CPU["value"]
valueGPU = GPU["value"]
standardErrorCPU = CPU["stdError"]
standardErrorGPU = GPU["stdError"]

for z in [10, 17] :
    plt.clf()
    exactValue = df.query("Engine == 'Analytical' & Type == 'EuropeanCall' ")["value"]
    exactValue = list(np.repeat(exactValue, z))
    nSimulations = CPU["nSimulations"]

    zipped_lists = zip(valueCPU, standardErrorCPU)
    under_line = [x - y for (x, y) in zipped_lists]
    zipped_lists = zip(valueCPU, standardErrorCPU)
    over_line = [x + y for (x, y) in zipped_lists]
    plt.plot(nSimulations[:z], valueCPU[:z], label = "Serial CPU", color='blue')
    plt.fill_between(nSimulations[:z], under_line[:z], over_line[:z], color='blue', alpha=.1)

    zipped_lists = zip(valueGPU, standardErrorGPU)
    under_line = [x - y for (x, y) in zipped_lists]
    zipped_lists = zip(valueGPU, standardErrorGPU)
    over_line = [x + y for (x, y) in zipped_lists]
    plt.plot(nSimulations[:z], valueGPU[:z], label = "Parallel GPU", color='orange')
    plt.fill_between(nSimulations[:z], under_line[:z], over_line[:z], color='orange', alpha=.1)


    plt.plot(nSimulations[:z], exactValue[:z], '-.',label = "Black-Sholes Analytical", color='green')

    plt.grid()
    plt.legend()
    plt.title("European option", fontsize=17)
    plt.ylabel("Price [€]", fontsize=14)
    plt.xlabel("Number of simulations", fontsize=14)

    plt.savefig('plot-' + str(z) + '.png')


    y = [15.5/math.sqrt(i) for i in nSimulations[:z] ]

    plt.clf()
    plt.plot(nSimulations[:z], y, label = "Expected error",color='green')
    plt.plot(nSimulations[:z], standardErrorCPU[:z], label = "Serial CPU",color='blue')
    plt.plot(nSimulations[:z], standardErrorGPU[:z], label = "Parallel GPU",color='orange')

    plt.grid()
    plt.legend()
    plt.title("European option - error", fontsize=17)
    plt.ylabel("Standard error", fontsize=14)
    plt.xlabel("Number of simulations", fontsize=14)

    plt.savefig('plotError-' + str(z) + '.png')