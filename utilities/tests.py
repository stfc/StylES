import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import os
import sys
import scipy as sc

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')

from parameters import *
from LES_constants import *
from LES_parameters import *
from LES_plot import *
from MSG_StyleGAN_tf2 import *

tf.random.set_seed(SEED_RESTART)



os.system("rm -rf tests")
os.system("mkdir tests")




#---------------------- verify orientation plots
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

A = np.asarray([[0, 1, 2], [3, 4, 1]])
#A = np.asarray([[[0, 1, 2], [3, 4, 1]], [[5, 6, 7], [8, 9, 6]], [[5, 6, 7], [8, 9, 6]], [[5, 6, 7], [8, 9, 6]]])

print(A.shape)
print(A)
print(A[1,0])
plt.pcolormesh(A)
plt.show()
plt.savefig("tests/pcolormesh.png")
B = cr(A, 0, 1)
print(B)
exit()




#--------------------------- compare filters

# N_DNS = 256
# L     = 50.176
# FILE_REAL = "/archive/jcastagna/Fields/HW/fields_N256_1image/fields_run21_time501.npz"


# # load numpy array
# U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
# U_DNS = np.cast[DTYPE](U_DNS)
# V_DNS = np.cast[DTYPE](V_DNS)
# P_DNS = np.cast[DTYPE](P_DNS)


# # plot DNS spectrum
# DELX  = L/N_DNS
# DELY  = L/N_DNS
# filename = "./results_tests/energy_spectrum_DNS.png"
# gradV = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
#               + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
# plot_spectrum(U_DNS, gradV, L, filename, close=False)


# # plot filtered fields using top-hat filter
# fU_DNS = U_DNS[::RS, ::RS]
# fV_DNS = V_DNS[::RS, ::RS]
# fP_DNS = P_DNS[::RS, ::RS]

# DELX  = L/N_DNS*RS
# DELY  = L/N_DNS*RS
# filename = "./results_tests/energy_spectrum_th.png"

# gradV = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*DELX))**2 \
#               + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*DELY))**2)
# plot_spectrum(fU_DNS, gradV, L, filename, close=False)



# # plot filtered fields using scipy gaussian
# fU_DNS = sc.ndimage.gaussian_filter(U_DNS, RS, mode='grid-wrap')
# fV_DNS = sc.ndimage.gaussian_filter(V_DNS, RS, mode='grid-wrap')
# fP_DNS = sc.ndimage.gaussian_filter(P_DNS, RS, mode='grid-wrap')

# fU_DNS = fU_DNS[::RS, ::RS]
# fV_DNS = fV_DNS[::RS, ::RS]
# fP_DNS = fP_DNS[::RS, ::RS]

# DELX  = L/N_DNS*RS
# DELY  = L/N_DNS*RS
# filename = "./results_tests/energy_spectrum_scg.png"

# gradV = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*DELX))**2 \
#               + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*DELY))**2)
# plot_spectrum(fU_DNS, gradV, L, filename, close=False)



# # plot filtered fields using scipy top-hat
# fU_DNS = sc.ndimage.white_tophat(U_DNS, RS, mode='grid-wrap')
# fV_DNS = sc.ndimage.white_tophat(V_DNS, RS, mode='grid-wrap')
# fP_DNS = sc.ndimage.white_tophat(P_DNS, RS, mode='grid-wrap')

# fU_DNS = fU_DNS[::RS, ::RS]
# fV_DNS = fV_DNS[::RS, ::RS]
# fP_DNS = fP_DNS[::RS, ::RS]

# DELX  = L/N_DNS*RS
# DELY  = L/N_DNS*RS
# filename = "./results_tests/energy_spectrum_scth.png"

# gradV = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*DELX))**2 \
#               + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*DELY))**2)
# plot_spectrum(fU_DNS, gradV, L, filename, close=False)



# # plot filtered fields using tf
# U_DNS = tf.convert_to_tensor(U_DNS)
# V_DNS = tf.convert_to_tensor(V_DNS)
# P_DNS = tf.convert_to_tensor(P_DNS)

# fU_DNS = gaussian_filter(U_DNS, rs=RS, rsca=RS)
# fV_DNS = gaussian_filter(V_DNS, rs=RS, rsca=RS)
# fP_DNS = gaussian_filter(P_DNS, rs=RS, rsca=RS)

# fU_DNS = fU_DNS[0,0,:,:].numpy()
# fV_DNS = fV_DNS[0,0,:,:].numpy()
# fP_DNS = fP_DNS[0,0,:,:].numpy()

# DELX  = L/N_DNS*RS
# DELY  = L/N_DNS*RS
# filename = "./results_tests/energy_spectrum_tf.png"

# gradV = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*DELX))**2 \
#               + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*DELY))**2)
# plot_spectrum(fU_DNS, gradV, L, filename, close=True)

# exit(0)





#--------------------------- create random tuning noise

# # parameters
# DTYPE     = "float32"
# NSIZE     = 512*17
# NC_NOISE  = 50
# NC2_NOISE = int(NC_NOISE/2)

# # initialization
# N2 = int(NC_NOISE/2)
# T  = NSIZE-1
# Dt = T/(NSIZE-1)
# t  = Dt*tf.cast(tf.random.uniform([NSIZE], maxval=NSIZE, dtype="int32"), DTYPE)
# t  = t[tf.newaxis,:]
# t  = tf.tile(t, [N2, 1])
# k  = tf.range(1,int(N2+1), dtype=DTYPE)
# f  = k/T
# f  = f[:,tf.newaxis]
# #c  = tf.constant(1.0, shape=[1,N2], dtype=DTYPE)

# # find noise
# for it in range(10):
#     phi    = tf.random.uniform([NC2_NOISE,1], minval=0.0, maxval=2.0*np.pi, seed=0, dtype=DTYPE) 
#     freq   = f * t
#     c      = tf.random.uniform([1,N2], minval=0, maxval=1.0, seed=0, dtype=DTYPE) 

#     argsin = tf.math.sin(2*np.pi*freq + phi)
#     noise  = tf.matmul(c,argsin)
#     noise  = 2.0 * (noise - tf.math.reduce_min(noise))/(tf.math.reduce_max(noise) - tf.math.reduce_min(noise)) - 1.0
#     noise  = noise - tf.math.reduce_mean(noise)

#     print(noise.shape)
#     count, bins, ignored = plt.hist(noise, 20, density=True)

#     ax = plt.gca()
#     plt.savefig("tests.png")
#     input()

# exit(0)    
    


# #--------------------------- create a tuning gaussian random noise
# N = 512

# a = -1*np.ones(N)
# b = np.ones(N)

# for i in range(100):
#     k = np.random.normal(loc=np.random.uniform(), scale=np.random.uniform(), size=N)
#     k = np.clip(k, 0, 1)

#     c = a*k + b*(1-k)
#     print(np.mean(a), np.mean(b), np.mean(c))
    
#     # count, bins, ignored = plt.hist(a, 15, density=True)
#     # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

#     # count, bins, ignored = plt.hist(b, 15, density=True)
#     # plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')

#     count, bins, ignored = plt.hist(c, 100, density=True)
#     plt.plot(bins, np.ones_like(bins), linewidth=2, color='y')
#     ax = plt.gca()
#     ax.set_ylim([0,1])
#     plt.savefig("tests.png")
#     input()
    
    
# N=16
# for i in range(10):
#     N=N*2
#     a = np.arange(N*N, dtype="float64")
#     a = np.reshape(a, [N,N])

#     tstart = time.time()
#     b = tf.convert_to_tensor(a)
#     print(str(N) + "CPU-GPU ", time.time() - tstart)

#     tstart = time.time()
#     c = b.numpy()
#     print(str(N) + "GPU-CPU ", time.time() - tstart)
    
#     print("\n")