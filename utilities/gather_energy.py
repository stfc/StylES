import numpy as np
from numpy.lib.npyio import savetxt
import os

list_te = [9,24,97,134]
N=10
kE = np.zeros((1024,2),dtype=np.float64)
for j, te in enumerate(list_te):
    kE[:,:] = 0
    for run in range(N):
        filename = "../LES_Solvers/energy/energy_run" + str(run) + "_" + str(te) + "te.txt"
        file = open(filename, 'r')
        lines = file.readlines()
        for i, line in enumerate(lines):
            val = line.split()
            k = val[0]
            E = val[1]
            kE[i,0] = kE[i,0] + float(k)
            kE[i,1] = kE[i,1] + float(E)

    kE = kE/N

    newfile = "../LES_Solvers/energy/average_" + str(te) + "te.txt"
    np.savetxt(newfile, kE, fmt='%.3e %.3e', newline=os.linesep)

