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


def find_vorticity(U, V):
    W = ((cr(V, 1, 0)-cr(V, -1, 0)) - (cr(U, 0, 1)-cr(U, 0, -1)))
    return W


def load_fields(filename='restart.npz', DNSrun=False):
    data = nc.load(filename)
    ctotTime = data['t']
    totTime = convert(ctotTime)
    U = data['U']
    V = data['V']
    P = data['P']
    if (DNSrun):
        C = data['C']
        B = data['B']
        return U, V, P, C, B, totTime
    else:
        return U, V, P, totTime


def save_fields(totTime, U, V, P, C=None, B=None, W=None, filename="restart.npz"):

    # save restart file
    nc.savez(filename, t=totTime, U=U, V=V, P=P, C=C, B=B, W=W)



def plot_spectrum(U, V, L, filename, close=True, label=None, xlim=[1e-2, 1e3], ylim=[1e-8, 1e1], useLogSca=True):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

    if useLogSca:
        plt.xscale("log")
        plt.yscale("log")

    # xLogLim    = [1.0e0, 1000]   # to do: to make nmore general
    # yLogLim    = [1.e-8, 0.1]
    # xLinLim    = [0.0e0, 600]
    # yLinLim    = [0.0e0, 1.0]
    # xLogLim    = [1.0e-1, 1.e+3]
    # yLogLim    = [1.e-11, 1.e+2]
    # xLinLim    = [0.0e0, 600]
    # yLinLim    = [0.0e0, 0.1]

    # plt.xlim(xlim)
    # plt.ylim(ylim) 

    if (label is not None):
        plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5, label=label)
        plt.legend()
    else:    
        plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5)
   

    if (close):
        #plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.savefig(filename, pad_inches=0)
        plt.close()

    filename = filename.replace(".png",".txt")
    print("knyquist for " + filename + " is:  " + str(knyquist))
    np.savetxt(filename, np.c_[wave_numbers, tke_spectrum], fmt='%1.4e')   # use exponential notation



def plot_spectrum_noPlots(U, V, L):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)
    return wave_numbers, tke_spectrum


def save_vel_violations(fname, uv_max, tstep, close=True):
    #save which plots have velocities U and V larger than 10
    #identifiers given by tail in LES_solver*
    arr = ["tstep is:"+str(tstep),uv_max[0], uv_max[1]]
    
    if (tstep == 0):
        f = open(fname, "w")
        np.savetxt(f, arr, newline='\n', fmt="%s")
    else:    
        f = open(fname, "a")
        np.savetxt(f, arr, newline='\n',  fmt="%s")

    if close:
        f.close()
