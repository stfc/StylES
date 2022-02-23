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


def save_fields(totTime, U, V, P, C, B, W, filename):

    # save restart file
    nc.savez("restart.npz", t=totTime, U=U, V=V, P=P, C=C, B=B, W=W)

    # save field for StyleGAN training
    maxU = np.max(U)
    minU = np.min(U)
    U_ = two*(U - minU)/(maxU- minU) - one

    maxV = np.max(V)
    minV = np.min(V)
    V_ = two*(V - minV)/(maxV - minV) - one

    maxP = np.max(P)
    minP = np.min(P)
    if (maxP!=minP):
        P_ = two*(P - minP)/(maxP - minP) - one
    else:
        P_ = P

    nc.savez(filename, U=U_, V=V_, P=P_)



def plot_spectrum(U, V, L, filename, close=False):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

    if useLogSca:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(xLogLim)
        plt.ylim(yLogLim)        
    else:
        plt.xlim(xLinLim)
        plt.ylim(yLinLim) 

    plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5)
    plt.savefig("Energy_spectrum.png", bbox_inches='tight', pad_inches=0)
    if (close):
        plt.close()

    np.savetxt(filename, np.c_[wave_numbers, tke_spectrum], fmt='%1.4e')   # use exponential notation



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
