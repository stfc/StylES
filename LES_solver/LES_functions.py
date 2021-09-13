import sys
import matplotlib.pyplot as plt

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../TurboGenPY/')

from tkespec import *
from tkespec import *
from cudaturbo import *

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

if (USE_CUPY):
    from cupy import sin, cos, sqrt, exp
else:
    from numpy import sin, cos, sqrt, exp


# wrapper for nc.roll
def cr(phi, i, j):
    return nc.roll(phi, (-i, -j), axis=(0,1))


def load_fields():
    data = nc.load('restart.npz')
    ctotTime = data['t']
    totTime = convert(ctotTime)
    U = data['U']
    V = data['V']
    P = data['P']
    C = data['C']
    B = data['B']

    return U, V, P, C, B, totTime


def save_fields(totTime, U, V, P, C, B):

    nc.savez('restart.npz', t=totTime, U=U, V=V, P=P, C=C, B=B)



def plot_spectrum(U, V, Lx, Ly, tstep):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, Lx, Ly, True)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5)
    plt.savefig("Energy_spectrum.png".format(tstep), bbox_inches='tight', pad_inches=0)
