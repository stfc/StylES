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

tf.random.set_seed(seed=SEED)

os.system("rm -rf results_tests")
os.system("mkdir results_tests")




# #---------------------- deep copy in numpy
# a = np.arange(3,5)
# #a = [3, 4]
# b = a
# c = a[:]
# d = a.copy()

# print(b is a) # True
# print(c is a) # False
# print(d is a) # False

# print(a, b, c, d) #[3 4] [3 4] [3 4] [3 4]

# a[0] = -11.

# print(a, b, c, d) #[-11   4] [-11   4] [-11   4] [3 4]


# U = np.arange(8)
# img_out = []
# for i in range(3):
#     res = 2**(i+1)
#     data = np.zeros([res], dtype=float)
#     U_t = U[::res]
#     data = U_t/8
#     U = np.arange(8,16)
#     img_out.append(data)

# print(img_out)
# exit()




# #---------------------- verify orientation plots
# def cr(phi, i, j):
#     return np.roll(phi, (-i, -j), axis=(0,1))

# A = np.asarray([[0, 1, 2], [3, 4, 1]])
# #A = np.asarray([[[0, 1, 2], [3, 4, 1]], [[5, 6, 7], [8, 9, 6]], [[5, 6, 7], [8, 9, 6]], [[5, 6, 7], [8, 9, 6]]])

# print(A.shape)
# print(A)
# print(A[1,0])
# plt.pcolormesh(A)
# plt.show()
# plt.savefig("tests/pcolormesh.png")
# B = cr(A, 0, 1)
# print(B)
# exit()




# #--------------------------- plot filtered quantities for IO
# L        = 50.176
# RS       = 16 # overwrite RS from parameters...
# RS2      = 16  # overwrite RS2 from parameters...
# DELX_LES = L/N_DNS*RS  # overwrite DELX_LES from parameters...
# DELY_LES = L/N_DNS*RS  # overwrite DELY_LES from parameters...
# filename_spec = "./results_tests/spectrumAll.png"

# # load numpy array
# U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
# U_DNS_org = np.cast[DTYPE](U_DNS)
# V_DNS_org = np.cast[DTYPE](V_DNS)
# P_DNS_org = np.cast[DTYPE](P_DNS)

# closePlot=False
# rs = 2
# for reslog in range(RES_LOG2, RES_LOG2-FIL-1, -1):
#     res = 2**(reslog+2)
#     data = np.zeros([3, res, res], dtype=DTYPE)
#     if (reslog==RES_LOG2):
#         fU_DNS = U_DNS_org
#         fV_DNS = V_DNS_org
#         fP_DNS = P_DNS_org
#     else:
#         if (TESTCASE=='mHW'):
#             fU_DNS = sc.ndimage.gaussian_filter(fU_DNS, rs, mode=['constant','wrap'])
#             fV_DNS = sc.ndimage.gaussian_filter(fV_DNS, rs, mode=['constant','wrap'])
#             fP_DNS = sc.ndimage.gaussian_filter(fP_DNS, rs, mode=['constant','wrap'])
#         else:
#             fU_DNS = sc.ndimage.gaussian_filter(fU_DNS, rs, mode='wrap')
#             fV_DNS = sc.ndimage.gaussian_filter(fV_DNS, rs, mode='wrap')
#             fP_DNS = sc.ndimage.gaussian_filter(fP_DNS, rs, mode='wrap')

#         fU_DNS = fU_DNS[::rs,::rs]
#         fV_DNS = fV_DNS[::rs,::rs]
#         fP_DNS = fP_DNS[::rs,::rs]

#     # normalize the data
#     minU = np.min(fU_DNS)
#     maxU = np.max(fU_DNS)
#     amaxU = max(abs(minU), abs(maxU))
#     if (amaxU<SMALL):
#         print("-----------Attention: invalid field!!!")
#         exit(0)
#     else:
#         fU_DNS = fU_DNS / amaxU
    
#     minV = np.min(fV_DNS)
#     maxV = np.max(fV_DNS)
#     amaxV = max(abs(minV), abs(maxV))
#     if (amaxV<SMALL):
#         print("-----------Attention: invalid field!!!")
#         exit(0)
#     else:
#         fV_DNS = fV_DNS / amaxV

#     minP = np.min(fP_DNS)
#     maxP = np.max(fP_DNS)
#     amaxP = max(abs(minP), abs(maxP))
#     if (amaxP<SMALL):
#         print("-----------Attention: invalid field!!!")
#         exit(0)
#     else:
#         fP_DNS = fP_DNS / amaxP

#     dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
#     dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)

#     if (reslog==RES_LOG2-FIL):
#         closePlot=True
#     plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename=filename_spec, label="differential_" + str(reslog), close=closePlot)

#     filename = "./results_tests/plot_FIL" + str(reslog) + ".png"
#     print_fields_3(fU_DNS, fV_DNS, fP_DNS, filename=filename, \
#         labels=['fU', 'fV', 'fP'], plot='same', testcase=TESTCASE) #, \
#         #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

# exit(0)






#--------------------------- compare filters
L        = 50.176
RS       = 8 # overwrite RS from parameters...
RSCA     = 1
DELX_LES = L/N_DNS*RS  # overwrite DELX_LES from parameters...
DELY_LES = L/N_DNS*RS  # overwrite DELY_LES from parameters...
filename = "./results_tests/filters_spectrum.png"

# load numpy array
U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
U_DNS = np.cast[DTYPE](U_DNS)
V_DNS = np.cast[DTYPE](V_DNS)
P_DNS = np.cast[DTYPE](P_DNS)

tU_DNS = tf.convert_to_tensor(U_DNS[tf.newaxis,tf.newaxis,:,:])
tV_DNS = tf.convert_to_tensor(V_DNS[tf.newaxis,tf.newaxis,:,:])
tP_DNS = tf.convert_to_tensor(P_DNS[tf.newaxis,tf.newaxis,:,:])


#--- DNS spectrum
dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX)
dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY)
plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename, label="DNS", close=False)


# # Top-hat (SciPy)
# fU_DNS = sc.ndimage.black_tophat(U_DNS, size=(N_DNS, N_DNS), mode='wrap')
# fV_DNS = sc.ndimage.black_tophat(V_DNS, size=(N_DNS, N_DNS), mode='wrap')
# fP_DNS = sc.ndimage.black_tophat(P_DNS, size=(N_DNS, N_DNS), mode='wrap')

# # fU_DNS = fU_DNS[::RS, ::RS]
# # fV_DNS = fV_DNS[::RS, ::RS]
# # fP_DNS = fP_DNS[::RS, ::RS]

# dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
# dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
# plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="Top-hat (SciPy)", close=False)

# fU_DNS_hsc = fU_DNS
# fV_DNS_hsc = fV_DNS
# fP_DNS_hsc = fP_DNS


#--- Gaussian (SciPy)
fU_DNS = sc.ndimage.gaussian_filter(U_DNS, sigma=RS, mode='wrap')
fV_DNS = sc.ndimage.gaussian_filter(V_DNS, sigma=RS, mode='wrap')
fP_DNS = sc.ndimage.gaussian_filter(P_DNS, sigma=RS, mode='wrap')

fU_DNS = fU_DNS[::RSCA,::RSCA]
fV_DNS = fV_DNS[::RSCA,::RSCA]
fP_DNS = fP_DNS[::RSCA,::RSCA]

dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="Gaussian (SciPy)", close=False)

fU_DNS_gsc = fU_DNS
fV_DNS_gsc = fV_DNS
fP_DNS_gsc = fP_DNS



#--- top-hat
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = define_filter(x_in[0,0,:,:], size=RS, rsca=RSCA, mean=0.0, delta=RS, type='Top-hat')
gfilter = tf.keras.Model(inputs=x_in, outputs=out)

fU_DNS = gfilter(tU_DNS)[0,0,:,:]
fV_DNS = gfilter(tV_DNS)[0,0,:,:]
fP_DNS = gfilter(tP_DNS)[0,0,:,:]

fU_DNS = fU_DNS.numpy()
fV_DNS = fV_DNS.numpy()
fP_DNS = fP_DNS.numpy()

dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="Top-hat", close=False)

fU_DNS_hat = fU_DNS
fV_DNS_hat = fV_DNS
fP_DNS_hat = fP_DNS


#--- Gaussian
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = define_filter(x_in[0,0,:,:], size=4*RS, rsca=RSCA, mean=0.0, delta=RS, type='Gaussian')
gfilter = tf.keras.Model(inputs=x_in, outputs=out)

fU_DNS = gfilter(tU_DNS)[0,0,:,:]
fV_DNS = gfilter(tV_DNS)[0,0,:,:]
fP_DNS = gfilter(tP_DNS)[0,0,:,:]

fU_DNS = fU_DNS.numpy()
fV_DNS = fV_DNS.numpy()
fP_DNS = fP_DNS.numpy()

dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="Gaussian", close=False)

fU_DNS_gtf = fU_DNS
fV_DNS_gtf = fV_DNS
fP_DNS_gtf = fP_DNS


#--- Spectral
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = define_filter(x_in[0,0,:,:], size=4*RS, rsca=RSCA, mean=0.0, delta=RS, type='Spectral')
gfilter = tf.keras.Model(inputs=x_in, outputs=out)

fU_DNS = gfilter(tU_DNS)[0,0,:,:]
fV_DNS = gfilter(tV_DNS)[0,0,:,:]
fP_DNS = gfilter(tP_DNS)[0,0,:,:]

fU_DNS = fU_DNS.numpy()
fV_DNS = fV_DNS.numpy()
fP_DNS = fP_DNS.numpy()

dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="Spectral", close=False)

fU_DNS_stf = fU_DNS
fV_DNS_stf = fV_DNS
fP_DNS_stf = fP_DNS


#--- Differential
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = define_filter(x_in[0,0,:,:], size=4*RS, rsca=RSCA, mean=0.0, delta=RS, type='Differential')
gfilter = tf.keras.Model(inputs=x_in, outputs=out)

fU_DNS = gfilter(tU_DNS)[0,0,:,:]
fV_DNS = gfilter(tV_DNS)[0,0,:,:]
fP_DNS = gfilter(tP_DNS)[0,0,:,:]

fU_DNS = fU_DNS.numpy()
fV_DNS = fV_DNS.numpy()
fP_DNS = fP_DNS.numpy()

dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="Differential", close=True)

fU_DNS_dtf = fU_DNS
fV_DNS_dtf = fV_DNS
fP_DNS_dtf = fP_DNS

exit(0)


# closePlot=False
# for i in range(10):
#     if i==0:
#         fU_DNS = tf.convert_to_tensor(U_DNS_org)
#         fV_DNS = tf.convert_to_tensor(V_DNS_org)
#         fP_DNS = tf.convert_to_tensor(P_DNS_org)
#         fU_DNS = fU_DNS[tf.newaxis,tf.newaxis,:,:]
#         fV_DNS = fV_DNS[tf.newaxis,tf.newaxis,:,:]
#         fP_DNS = fP_DNS[tf.newaxis,tf.newaxis,:,:]

#     fU_DNS = gfilter(fU_DNS)[0,0,:,:]
#     fV_DNS = gfilter(fV_DNS)[0,0,:,:]
#     fP_DNS = gfilter(fP_DNS)[0,0,:,:]

#     nU_DNS = fU_DNS.numpy()
#     nV_DNS = fV_DNS.numpy()
#     nP_DNS = fP_DNS.numpy()

#     gradV = np.sqrt(((cr(nV_DNS, 1, 0) - cr(nV_DNS, -1, 0))/(2.0*DELX_LES))**2 \
#                   + ((cr(nV_DNS, 0, 1) - cr(nV_DNS, 0, -1))/(2.0*DELY_LES))**2)
#     if i==9:
#         closePlot=True
#     plot_spectrum_2d_3v(nU_DNS, gradV, L, filename, label="differential_" + str(i), close=closePlot)

#     # filename = "./results_tests/filters_plot_diff_" + str(i) + ".png"
#     # print_fields_3(nU_DNS, nV_DNS, nP_DNS, filename=filename, \
#     #     labels=['diff U', 'diff V', 'diff P'], plot='same', testcase=TESTCASE) #, \
#     #     #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)
        
#     print("Done for i ", i)


# verify differential filter
alpha = np.linspace(0,10,1)
for i,al in enumerate(alpha):
    # fP_DNS_rec = fP_DNS_dtf - al*(-tr(fP_DNS_dtf, 2, 0) + 16*tr(fP_DNS_dtf, 1, 0) - 30*fP_DNS_dtf + 16*tr(fP_DNS_dtf,-1, 0) - tr(fP_DNS_dtf,-2, 0))/(12*DELX**2) \
    #                         - al*(-tr(fP_DNS_dtf, 0, 2) + 16*tr(fP_DNS_dtf, 0, 1) - 30*fP_DNS_dtf + 16*tr(fP_DNS_dtf, 0,-1) - tr(fP_DNS_dtf, 0,-2))/(12*DELY**2)
    fP_DNS_rec = fP_DNS_dtf - al*(tr(fP_DNS_dtf, 1, 0) - 2*fP_DNS_dtf + tr(fP_DNS_dtf,-1, 0))/(DELX**2) \
                            - al*(tr(fP_DNS_dtf, 0, 1) - 2*fP_DNS_dtf + tr(fP_DNS_dtf, 0,-1))/(DELY**2)

    res = tf.reduce_sum(tf.math.squared_difference(P_DNS_org[::RS,::RS], fP_DNS_rec)).numpy()
    if (i==0):
        minRes = res
        almin = al
    else:
        if (res<minRes):
            minRes = res
            almin = al
            print(i, al, res)

print(almin)           

fU_DNS_rec = fU_DNS_dtf - almin*(-tr(fU_DNS_dtf, 2, 0) + 16*tr(fU_DNS_dtf, 1, 0) - 30*fU_DNS_dtf + 16*tr(fU_DNS_dtf,-1, 0) - tr(fU_DNS_dtf,-2, 0))/(12*DELX**2) \
                        - almin*(-tr(fU_DNS_dtf, 0, 2) + 16*tr(fU_DNS_dtf, 0, 1) - 30*fU_DNS_dtf + 16*tr(fU_DNS_dtf, 0,-1) - tr(fU_DNS_dtf, 0,-2))/(12*DELY**2)

fV_DNS_rec = fV_DNS_dtf - almin*(-tr(fV_DNS_dtf, 2, 0) + 16*tr(fV_DNS_dtf, 1, 0) - 30*fV_DNS_dtf + 16*tr(fV_DNS_dtf,-1, 0) - tr(fV_DNS_dtf,-2, 0))/(12*DELX**2) \
                        - almin*(-tr(fV_DNS_dtf, 0, 2) + 16*tr(fV_DNS_dtf, 0, 1) - 30*fV_DNS_dtf + 16*tr(fV_DNS_dtf, 0,-1) - tr(fV_DNS_dtf, 0,-2))/(12*DELY**2)

fP_DNS_rec = fP_DNS_dtf - almin*(-tr(fP_DNS_dtf, 2, 0) + 16*tr(fP_DNS_dtf, 1, 0) - 30*fP_DNS_dtf + 16*tr(fP_DNS_dtf,-1, 0) - tr(fP_DNS_dtf,-2, 0))/(12*DELX**2) \
                        - almin*(-tr(fP_DNS_dtf, 0, 2) + 16*tr(fP_DNS_dtf, 0, 1) - 30*fP_DNS_dtf + 16*tr(fP_DNS_dtf, 0,-1) - tr(fP_DNS_dtf, 0,-2))/(12*DELY**2)


filename = "./results_tests/filters_plot_diff_U.png"
print_fields_3(U_DNS_org[::RS,::RS], fU_DNS_rec, U_DNS_org[::RS,::RS]-fU_DNS_rec, filename=filename, \
    labels=['DNS U', 'reconstructed U', 'diff'], plot='diff', testcase=TESTCASE) #, \
    #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

filename = "./results_tests/filters_plot_diff_V.png"
print_fields_3(V_DNS_org[::RS,::RS], fV_DNS_rec, V_DNS_org[::RS,::RS]-fV_DNS_rec, filename=filename, \
    labels=['DNS V', 'reconstructed V', 'diff'], plot='diff', testcase=TESTCASE) #, \
    #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

filename = "./results_tests/filters_plot_diff_P.png"
print_fields_3(P_DNS_org[::RS,::RS], fP_DNS_rec, P_DNS_org[::RS,::RS]-fP_DNS_rec, filename=filename, \
    labels=['DNS P', 'reconstructed P', 'diff'], plot='diff', testcase=TESTCASE) #, \
    #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)





filename = "./results_tests/filters_plot_U.png"
print_fields_3(fU_DNS_stf, fU_DNS_gtf, fU_DNS_dtf, filename=filename, labels=['spectral U', 'gaussian U', 'differential U'], plot='same', testcase=TESTCASE) #, \
    #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

filename = "./results_tests/filters_plot_V.png"
print_fields_3(fV_DNS_stf, fV_DNS_gtf, fV_DNS_dtf, filename=filename, labels=['spectral V', 'gaussian V', 'differential V'], plot='same', testcase=TESTCASE) #, \
    #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

filename = "./results_tests/filters_plot_P.png"
print_fields_3(fP_DNS_stf, fP_DNS_gtf, fP_DNS_dtf, filename=filename, labels=['spectral P', 'gaussian P', 'differential P'], plot='same', testcase=TESTCASE) #, \
    #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

exit(0)





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



# print("TensorFlow version:", tf.__version__)


# tf.config.run_functions_eagerly(True)

# RAND_NOISE = True

# class layer_dummy(layers.Layer):
#     def __init__(self, x, **kwargs):
#         super(layer_dummy, self).__init__(**kwargs)

#         c_init = tf.ones_initializer()
#         self.c = tf.Variable(
#             initial_value=c_init(shape=[x.shape[-2],x.shape[-1]]),
#             trainable=False,
#             name="dummy_var",
#         )

#     def call(self, vin, xshape):
#         x = vin + tf.reshape(self.c, shape=xshape)
#         return x



# def make_dummy_model():
#     vin = tf.keras.Input(shape=([1, 10]))

#     ldummy = layer_dummy(vin)
#     vout = ldummy(vin, [vin.shape[-2],vin.shape[-1]])

#     dummy_model = Model(inputs=vin, outputs=vout)
#     return dummy_model


# dummy = make_dummy_model()
# dummy.summary()

# exit()











# #-------------------------- test: noise generation---------------------------- 
# DTYPE = "float32"
# LATENT_SIZE = 512
# N  = 100
# N2 = int(N/2)

# zl_init = tf.ones_initializer()
# c = tf.Variable(
#     initial_value=zl_init(shape=[1,N2], dtype="float32"),
#     trainable=True,
#     name="zlatent_k1"
# )

# # zl_init = tf.range(-250,250,dtype="float32")
# # z2 = zl_init #*zl_init
# # z2 = z2[tf.newaxis,:]
# # c = tf.Variable(z2,
# #     trainable=True,
# #     name="zlatent_k1"
# # )

# T  = LATENT_SIZE-1
# Dt = T/(LATENT_SIZE-1)
# t  = Dt*tf.range(LATENT_SIZE, dtype="float32")
# t  = t[tf.newaxis,:]
# t  = tf.tile(t, [N2, 1])
# k  = tf.range(1,int(N2+1), dtype=DTYPE)
# f  = k/T
# f  = f[:,tf.newaxis]

# phi = tf.random.uniform([N2,1], maxval=2.0*np.pi, dtype=DTYPE)
# #phi = tf.ones([N2,1])

# freq = f * t

# argsin = tf.math.sin(2*np.pi*freq + phi)
# x = tf.matmul(c,argsin)

# minRan = tf.math.reduce_min(x)
# x = x - minRan

# plt.plot(x[0,:].numpy())
# plt.savefig('dummy_test1.png')
# plt.close()

# plt.hist(x[0,:].numpy(), 50)
# plt.savefig('dummy_test2.png')





# DTYPE = "float32"
# LATENT_SIZE = 512
# NC_NOISE = 100
# NC2_NOISE = int(NC_NOISE/2)
# DTYPE    = "float32"

# class layer_noise(tf.keras.layers.Layer):
#     def __init__(self, x, ldx, **kwargs):
#         super(layer_noise, self).__init__(**kwargs)

#         self.NSIZE = x.shape[-2] * x.shape[-1]
#         self.N     = NC_NOISE
#         self.N2    = int(self.N/2)
#         self.T     = self.NSIZE-1
#         self.Dt    = self.T/(self.NSIZE-1)
#         self.t     = self.Dt*tf.range(self.NSIZE, dtype=DTYPE)
#         self.t     = self.t[tf.newaxis,:]
#         self.t     = tf.tile(self.t, [self.N2, 1])
#         self.k     = tf.range(1,int(self.N2+1), dtype=DTYPE)
#         self.f     = self.k/self.T
#         self.f     = self.f[:,tf.newaxis]

#         c_init = tf.ones_initializer()
#         self.c = tf.Variable(
#             initial_value=c_init(shape=[1,self.N2], dtype=DTYPE),
#             trainable=True,
#             name="noise_%d" % ldx
#         )

#     def call(self, x, phi):

#         freq = self.f * self.t
#         argsin = tf.math.sin(2*np.pi*freq + phi)
#         noise = tf.matmul(self.c,argsin)
#         noise = tf.reshape(noise, shape=x.shape)

#         return noise


# style_in = tf.ones(shape=[1, LATENT_SIZE], dtype=DTYPE)
# #phi      = tf.ones([NC2_NOISE,1], dtype=DTYPE)
# phi      = tf.random.uniform([NC2_NOISE,1], maxval=2.0*np.pi, dtype=DTYPE)
# lnoise   = layer_noise(style_in, 0)
# z        = lnoise(style_in, phi)

# filename = "z_latent.png"
# plt.plot(z.numpy()[0,:])
# plt.savefig(filename)
# plt.close()




# LATENT_SIZE = 512
# N   = 10
# N2  = int(N/2)
# T   = LATENT_SIZE-1
# Dt  = T/(LATENT_SIZE-1)
# t   = Dt*np.arange(1,LATENT_SIZE)
# k   = np.arange(1,int(N2))
# c   = np.ones(N2)
# phi = np.random.uniform(0,2*np.pi,N2)
# f   = k/T
# C0  = 0

# v = C0/2*np.ones(LATENT_SIZE)
# for i in range(LATENT_SIZE):
#     for k in range(1,N2):
#         v[i-1] = v[i-1] + c[k-1]*np.sin(2*np.pi*f[k-1]*t[i-1] + phi[k-1])

# plt.plot(v)
# plt.savefig('dummy_test1.png')






# for k in range(1):
#     filename = "results_reconstruction/wf_" + str(k) + ".npz"
#     data = np.load(filename)
#     wf = data['wf']
#     print(wf.shape)
#     for j in range(10,11):
#         plt.plot(wf[0,j,:])

#     plt.savefig('test1.png', linewidth=0.01)

# exit()

# a = tf.random.uniform([3,1],maxval=5, dtype="int32")
# b = tf.random.uniform([1,3,2],maxval=5, dtype="int32")
# c = a[:,:]*b[:,:,:]
# print(a)
# print(b)
# print(c)

# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model, layers


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Add a channels dimension
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000).batch(32)

# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# #-------------Layer Noise
# def apply_noise(x):
#     w_init = tf.zeros_initializer()
#     weight = tf.Variable(
#     initial_value=w_init(shape=[1]),
#     trainable=True,
#     )
#     return x*weight


# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     x = self.d2(x)
#     x = apply_noise(x)
#     return x



# # Create an instance of the model
# model = MyModel()

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# optimizer = tf.keras.optimizers.Adam()


# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# #@tf.function
# def train_step(images, labels):
#   with tf.GradientTape() as tape:
#     # training=True is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions = model(images, training=True)
#     loss = loss_object(labels, predictions)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#   train_loss(loss)
#   train_accuracy(labels, predictions)



# #@tf.function
# def test_step(images, labels):
#   # training=False is only needed if there are layers with different
#   # behavior during training versus inference (e.g. Dropout).
#   predictions = model(images, training=False)
#   t_loss = loss_object(labels, predictions)

#   test_loss(t_loss)
#   test_accuracy(labels, predictions)




# EPOCHS = 1

# for epoch in range(EPOCHS):
#   # Reset the metrics at the start of the next epoch
#   train_loss.reset_states()
#   train_accuracy.reset_states()
#   test_loss.reset_states()
#   test_accuracy.reset_states()

#   for images, labels in train_ds:
#     train_step(images, labels)

#   for test_images, test_labels in test_ds:
#     test_step(test_images, test_labels)

#   print(
#     f'Epoch {epoch + 1}, '
#     f'Loss: {train_loss.result()}, '
#     f'Accuracy: {train_accuracy.result() * 100}, '
#     f'Test Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )

# model.summary()




  
# # x = tf.Variable(3.0)
# # c = tf.Variable(4.0)

# with tf.GradientTape() as tape:
#   x = c*5
#   y = x**2

# dx_dc = tape.gradient(x, c)

# print(dx_dc.numpy())





# layer = tf.keras.layers.Dense(2, activation='relu')
# c = tf.constant([[1., 2., 3.]])
# x = tf.Variable([[1., 2., 3.]], trainable=True)

# with tf.GradientTape() as tape:
#   # Forward pass
#   x = c*2
#   y = layer(x)
#   loss = tf.reduce_mean(y**2)

# # Calculate gradients with respect to every trainable variable
# grad = tape.gradient(loss, layer.trainable_variables)


# for var, g in zip(layer.trainable_variables, grad):
#   print(f'{var.name}, shape: {g.shape}')






# mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)






# #-------------Layer Noise
# class layer_noise(layers.Layer):
#     def __init__(self, **kwargs):
#         super(layer_noise, self).__init__(**kwargs)

#         w_init = tf.ones_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=[1]),
#             trainable=False,
#             **kwargs
#         )

#     def call(self, x):
#         return tf.cast(self.w, x.dtype)

# # def apply_noise(x):
# #     # w_init = tf.zeros_initializer()
# #     # weight = tf.Variable(
# #     # initial_value=w_init(shape=[1]),
# #     # trainable=False,
# #     # )
# #     # return x*weight
# #     #lnoise = layer_noise(x,name="layer_noise")
# #     nweights = lnoise(x)
# #     return x*nweights


# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)
#     self.n1 = layer_noise(name="layer_noise")
#     #self.var = tf.Variable(2., trainable=False)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     x = self.d2(x)
#     w = self.n1(x)
#     x = x*w
#     return x
