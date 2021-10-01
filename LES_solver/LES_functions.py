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


# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../TurboGenPY/')

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
    #filename = "restart_" + str(it) + ".npz"
    nc.savez(filename, t=totTime, U=U, V=V, P=P, C=C, B=B)



def plot_spectrum(U, V, L, tstep):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5)
    plt.savefig("Energy_spectrum.png".format(tstep), bbox_inches='tight', pad_inches=0)

    filename = "Energy_spectrum_" + str(tstep) + ".txt"
    np.savetxt(filename, np.c_[wave_numbers, tke_spectrum], fmt='%1.4e')   # use exponential notation
