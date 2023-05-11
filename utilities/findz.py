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
import os
import sys
from xml.dom import minidom
import scipy as sc

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *
from HIT_2D import L

os.chdir('../')
from MSG_StyleGAN_tf2 import *
os.chdir('./utilities')

tf.random.set_seed(SEED_RESTART)


# local parameters
NL             = 10000       # number of different latent vectors randomly selected
#FILE_REAL     = "../../../data/HW_N512_reconstruction/fields/fields_run1000_time200.npz"
FILE_REAL      = "../LES_Solvers/fields/fields_run0_it9000.npz"
N_DNS          = 2**RES_LOG2
N2_DNS         = int(N_DNS/2.0)
tollFIL        = 1.e-4
CALC_VORTICITY = False

# loading StyleGAN checkpoint
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")

time.sleep(3)


# load fields
U_DNS_org, V_DNS_org, P_DNS_org, totTime = load_fields(FILE_REAL)
U_DNS_org = np.cast[DTYPE](U_DNS_org)
V_DNS_org = np.cast[DTYPE](V_DNS_org)
P_DNS_org = np.cast[DTYPE](P_DNS_org)

if (TESTCASE=='HIT_2D'):
    P_DNS_org = find_vorticity(U_DNS_org, V_DNS_org)

# normalize
U_DNS_org = 2.0*(U_DNS_org - np.min(U_DNS_org))/(np.max(U_DNS_org) - np.min(U_DNS_org)) - 1.0
V_DNS_org = 2.0*(V_DNS_org - np.min(V_DNS_org))/(np.max(V_DNS_org) - np.min(V_DNS_org)) - 1.0
P_DNS_org = 2.0*(P_DNS_org - np.min(P_DNS_org))/(np.max(P_DNS_org) - np.min(P_DNS_org)) - 1.0

# plot centerlin
plt.plot(U_DNS_org[N2_DNS,:], color='black', label='org')
plt.legend()

savei = 0
minDiff = large
for i in range(NL):

    # find new fields
    zlatents    = tf.random.uniform([1,NC2_NOISE,1], maxval=2.0*np.pi, dtype=DTYPE, seed=SEED)
    wlatents, _ = mapping(zlatents, training=False)
    predictions = synthesis(wlatents, training=False)
    UVP_DNS = predictions[RES_LOG2-2]

    # find DNS from GAN fields
    U_DNS = UVP_DNS[0, 0, :, :].numpy()
    V_DNS = UVP_DNS[0, 1, :, :].numpy()
    P_DNS = UVP_DNS[0, 2, :, :].numpy()

    if (TESTCASE=='HIT_2D'):
        P_DNS = find_vorticity(U_DNS, V_DNS)

    # normalize
    U_DNS = 2.0*(U_DNS - np.min(U_DNS))/(np.max(U_DNS) - np.min(U_DNS)) - 1.0
    V_DNS = 2.0*(V_DNS - np.min(V_DNS))/(np.max(V_DNS) - np.min(V_DNS)) - 1.0
    P_DNS = 2.0*(P_DNS - np.min(P_DNS))/(np.max(P_DNS) - np.min(P_DNS)) - 1.0


    # find difference with target image
    diff = tf.math.reduce_mean(tf.math.squared_difference(UVP_DNS[0,0,N2_DNS,:], U_DNS_org[N2_DNS,:]))

    # swap and plot if found a new minimum
    if (diff < minDiff):
        minDiff = diff
        savei = i
        print(i, minDiff.numpy())

        plt.plot(UVP_DNS[0,0,N2_DNS,:], label=str(i) + " " + str(minDiff.numpy()))
        plt.legend()
        plt.savefig("findz.png")

        filename = "findz_fields_diff.png"
        print_fields_3(P_DNS_org, UVP_DNS[0,2,:,:], P_DNS_org-UVP_DNS[0,2,:,:], N=N_DNS, filename=filename, \
        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0, diff=True)

    if (i%100 == 0):
        print ("done for i ", i)


