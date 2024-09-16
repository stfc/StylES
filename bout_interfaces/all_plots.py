import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')

from parameters import *
from LES_plot import *
from LES_functions import *

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d_3v
from isoturb import generate_isotropic_turbulence_2d


plt.rcParams['mathtext.default'] = 'regular'
plt.matplotlib.rcParams.update({'font.size': 16})
plt.matplotlib.rcParams.update({'figure.autolayout': True})


tailist = [[r"HW $256^2$", "HW_N256"], [r"HW $512^2$", "HW_N512"], [r"mHW $512^2$", "mHW_N512"]]
for tail, ntail in tailist:
    file_path = ntail +"/energy_vs_time.npz"
    data = np.load(file_path)
    tD   = data['tD']
    eD   = data['eD']
    tS   = data['tS']
    eS   = data['eS']
    plt.yscale("log")
    plt.plot(tD, eD, label="energy DNS",    color='k', linestyle='solid')
    plt.plot(tS, eS, label="energy StylES", color='k', linestyle='dotted')
    
    file_path = ntail +"/enstrophy_vs_time.npz"
    data = np.load(file_path)
    tD   = data['tD']
    eD   = data['eD']
    tS   = data['tS']
    eS   = data['eS']
    plt.yscale("log")
    plt.plot(tD, eD, label="enstrophy DNS",    color='r', linestyle='solid')
    plt.plot(tS, eS, label="enstrophy StylES", color='r', linestyle='dotted')

    file_path = ntail +"/radialFlux_vs_time.npz"
    data = np.load(file_path)
    tD   = data['tD']
    eD   = data['eD']
    tS   = data['tS']
    eS   = data['eS']
    plt.yscale("log")
    plt.plot(tD, np.absolute(eD), label="radial flux DNS",    color='g', linestyle='solid')
    plt.plot(tS, np.absolute(eS), label="radial flux StylES", color='g', linestyle='dotted')

    file_path = ntail +"/poloidalFlux_vs_time.npz"
    data = np.load(file_path)
    tD   = data['tD']
    eD   = data['eD']
    tS   = data['tS']
    eS   = data['eS']
    plt.yscale("log")
    plt.plot(tD, np.absolute(eD), label="poloidal flux DNS",    color='b', linestyle='solid')
    plt.plot(tS, np.absolute(eS), label="poloidal flux StylES", color='b', linestyle='dotted')

    plt.xlim(0,2.5)
    plt.xticks(np.arange(0,2.6,0.5))
    plt.xlabel("time [$\omega_{ci}^{-1}$]")
    plt.ylabel(r"$E$, $\xi$, $|\Gamma_r|$, $|\Gamma_p|$")
    # plt.legend(frameon=False, fontsize="10", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig("values_" + ntail + ".png", dpi=300)
    plt.close()