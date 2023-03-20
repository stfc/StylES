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
import scipy as sc

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *

os.chdir('../')
from MSG_StyleGAN_tf2 import *
os.chdir('./utilities')





N_DNS = 2**RES_LOG2
N_LES = 2**RES_LOG2_FIL
zero_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)

# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)



tf.random.set_seed(SEED_RESTART)
zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
dlatents = mapping(zlatents, training=False)


# verify
# data = np.load("../LES_Solvers/restart.npz")
# latent = data['newl']
# dlatents = tf.tile(latent[np.newaxis, :], [1, RES_LOG2*2-2, 1])

# print(latent)

# inference
predictions = synthesis(dlatents, training=False)



# write fields and energy spectra for each layer
UVP_DNS = predictions[RES_LOG2-2]

U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()
if (TESTCASE=='HIT_2D'):
    P_DNS = find_vorticity(U_DNS, V_DNS)

U_DNS = 2.0*(U_DNS - tf.math.reduce_min(U_DNS))/(tf.math.reduce_max(U_DNS) - tf.math.reduce_min(U_DNS)) - 1.0
V_DNS = 2.0*(V_DNS - tf.math.reduce_min(V_DNS))/(tf.math.reduce_max(V_DNS) - tf.math.reduce_min(V_DNS)) - 1.0
P_DNS = 2.0*(P_DNS - tf.math.reduce_min(P_DNS))/(tf.math.reduce_max(P_DNS) - tf.math.reduce_min(P_DNS)) - 1.0

if (TESTCASE=='HIT_2D'):
    filename = "../LES_Solvers/plots_restart.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, OUTPUT_DIM, filename, TESTCASE, \
        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

    filename = "../LES_Solvers/restart"
    save_fields(0, U_DNS, V_DNS, zero_DNS, zero_DNS, zero_DNS, P_DNS, filename)

    filename = "../LES_Solvers/energy_spectrum_restart.png"
    closePlot=True
    plot_spectrum(U_DNS, V_DNS, L, filename, close=closePlot)

else:
    print("No restart created!")

exit(0)
