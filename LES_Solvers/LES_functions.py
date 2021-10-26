import sys
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

if (USE_GPU):
    from cupy import sin, cos, sqrt, exp
else:
    from numpy import sin, cos, sqrt, exp


# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../TurboGenPY/')

from tkespec import compute_tke_spectrum2d
from isoturb import generate_isotropic_turbulence_2d




# wrapper for nc.roll
def cr(phi, i, j):
    return nc.roll(phi, (-i, -j), axis=(0,1))


def load_fields(filename='restart.npz'):
    data = nc.load(filename)
    ctotTime = data['t']
    totTime = convert(ctotTime)
    U = data['U']
    V = data['V']
    P = data['P']
    C = data['C']
    B = data['B']

    return U, V, P, C, B, totTime


def save_fields(totTime, it, U, V, P, C, B):

    filename = "restart.npz"
    nc.savez(filename, t=totTime, U=U, V=V, P=P, C=C, B=B)

    filename = "fields_it{0:d}.npz".format(it)
    maxU = np.max(U)
    maxV = np.max(V)
    minU = np.min(U)
    minV = np.min(V)
    maxVel = max(maxU, maxV)
    minVel = min(minU, minV)
    U_ = (U-minVel)/(maxVel-minVel)
    V_ = (V-minVel)/(maxVel-minVel)
    W = (cr(U, 0, 1)-cr(U, 0, -1)) - (cr(V, 1, 0)-cr(V, -1, 0))
    W = convert(W)
    nc.savez(filename, U=U_, V=V_)



def plot_spectrum(U, V, L, it, name="", close=False):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

    plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5)
    plt.savefig("Energy_spectrum.png", bbox_inches='tight', pad_inches=0)
    if (close):
        plt.close()

    filename = "Energy_spectrum_it{0:d}.txt".format(it)
    np.savetxt(filename, np.c_[wave_numbers, tke_spectrum], fmt='%1.4e')   # use exponential notation
