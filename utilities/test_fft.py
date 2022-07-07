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
import matplotlib.pyplot as plt

from scipy.fft import fftn

N = 20
x = np.arange(0.0,2.0*np.pi+2.0*np.pi/(N-1),2.0*np.pi/(N-1))
y = np.cos(1*x)

print(x[N-1])

plt.plot(x,y)
plt.savefig("test.png")
plt.close()

yf = fftn(y)

for i in range(len(yf)):
    print(i, int(yf[i].real))

plt.plot(yf)
plt.savefig("test2.png")