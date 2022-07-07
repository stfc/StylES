#----------------------------------------------------------------------------------------------
#
#    Copyright (C): 2022 UKRI-STFC (Hartree Centre)
#
#    Author: Jony Castagna, Francesca Schiavello
#
#    Licence: This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------------------------------
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

