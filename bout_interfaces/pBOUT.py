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


#------------------------------------------------------ initialize StylES procedure

print("\n\n------------------------------------- Start StylES initialization -------------")

import os
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
import sys

RUN_TEST = False
if (RUN_TEST):
    PATH_StylES = "../"
else:
    PATH_StylES = "../../../../StylES/"

sys.path.insert(0, PATH_StylES + './')
sys.path.insert(0, PATH_StylES + './LES_Solvers/')
sys.path.insert(0, PATH_StylES + '../TurboGenPY/')


from LES_constants import *
from LES_parameters import *
from LES_plot import *

from parameters       import *
from MSG_StyleGAN_tf2 import *
from functions        import *

from pyevtk.hl import gridToVTK





#------------------------------------------------------ define local parameters
TUNE         = False
TUNE_NOISE   = False
RELOAD_FREQ  = 10000
delx         = 1.0
dely         = 1.0
delx_LES     = 1.0
dely_LES     = 1.0
tollLES      = 0.25
CHKP_DIR     = PATH_StylES + "checkpoints/"
CHKP_DIR_WL  = PATH_StylES + "bout_interfaces/restart_fromGAN/checkpoints_wl/"
LES_pass     = lr_DNS_maxIt
pPrintFreq   = 0.01
RUN_DNS      = False
RESTART_WL   = True
USE_DIFF_LES = False 
IMPLICIT     = False  
PROFILE_BOUT = False
SIZE         = N_LES*BATCH_SIZE*N_LES

tf.random.set_seed(seed=SEED_RESTART)

#------------------------------------------------------ define initial quantities, model, optimizer, fields in
# clean up and prepare folders
os.system("rm -rf results_StylES")
os.system("mkdir -p results_StylES/fields")
os.system("mkdir -p " + CHKP_DIR_WL)

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)

if (DIMS_3D):
    BOUT_U_LES  = np.zeros((N_LES,BATCH_SIZE,N_LES), dtype=DTYPE)
    BOUT_V_LES  = np.zeros((N_LES,BATCH_SIZE,N_LES), dtype=DTYPE)
    BOUT_P_LES  = np.zeros((N_LES,BATCH_SIZE,N_LES), dtype=DTYPE)
    BOUT_F_LES  = np.zeros((N_LES,BATCH_SIZE,N_LES), dtype=DTYPE)
    BOUT_G_LES  = np.zeros((N_LES,BATCH_SIZE,N_LES), dtype=DTYPE)
else:    
    BOUT_U_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
    BOUT_V_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
    BOUT_P_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
    BOUT_F_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
    BOUT_G_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)


# define optimizer for z and w search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")




# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = apply_filter_NCH(x_in, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian')
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_in_single    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out_single     = apply_filter_NCH(x_in_single, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian', NCH=1)
    gfilter_single = tf.keras.Model(inputs=x_in_single, outputs=out_single)

    x_in        = tf.keras.Input(shape=([1, RS+1, RS+1]), dtype=DTYPE)
    out         = apply_filter(x_in[0,0,:,:], size=4*RS, rsca=1, mean=0.0, delta=RS, subsection=True, type='Gaussian')
    gfilter_sub = tf.keras.Model(inputs=x_in, outputs=out)
else:
    gfilter     = filters[IFIL]



# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    
print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)



#--------------------------- restart from defined values
LES_all0 = []
if (RESTART_WL):

    filename = PATH_StylES + "/bout_interfaces/restart_fromGAN/z0.npz"
                
    data = np.load(filename)

    z0         = data["z0"]
    dlatents   = data["dlatents"]
    LES_in0    = data["LES_in0"]
    nUVP_amaxo = data["nUVP_amaxo"]
    fUVP_amaxo = data["fUVP_amaxo"]
    
    print("z0",                 z0.shape, np.min(z0),         np.max(z0))
    print("dlatents",     dlatents.shape, np.min(dlatents),   np.max(dlatents))
    print("LES_in0",       LES_in0.shape, np.min(LES_in0),    np.max(LES_in0))
    print("nUVP_amaxo", nUVP_amaxo.shape, np.min(nUVP_amaxo), np.max(nUVP_amaxo))
    print("fUVP_amaxo", fUVP_amaxo.shape, np.min(fUVP_amaxo), np.max(fUVP_amaxo))        

    # assign variables
    z0         = tf.convert_to_tensor(z0, dtype=DTYPE)
    LES_in0    = tf.convert_to_tensor(LES_in0, dtype=DTYPE)

    UVP_max = np.concatenate([nUVP_amaxo, fUVP_amaxo], 0)

    # assign variable noise
    if (TUNE_NOISE):
        noise_DNS = data["noise_DNS"]
        print("noise_DNS", noise_DNS.shape, np.min(noise_DNS), np.max(noise_DNS))
        noise_DNS = tf.convert_to_tensor(noise_DNS,  dtype=DTYPE)
        it=0
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                layer.trainable_variables[0].assign(noise_DNS[it])
                it=it+1

    # set LES_in fields
    for res in range(2,RES_LOG2-FIL):
        rs = 2**(RES_LOG2-FIL-res)
        LES_all0.append(LES_in0[:,:,::rs,::rs])

else:

    # set z
    if (DIMS_3D):
        z0a = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        z0b = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        z0 = z0a
        for i in range(1,BATCH_SIZE):
            wi = np.cos(i/float(BATCH_SIZE-1)*2.0*np.pi)
            zi = z0a*wi + z0b*(1.0-wi)
            z0 = tf.concat([z0, zi], axis=0)
    else:
        z0 = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        
    dlatents = mapping(z0, training=False)

    if (not USE_IMGSLES):
        pre_img  = pre_synthesis(dlatents, training = False)
        LES_in0  = pre_img[-1] 

    for res in range(2,RES_LOG2-FIL+1):
        rs = 2**(RES_LOG2-FIL-res)
        if (res != RES_LOG2-FIL):
            LES_all0.append(LES_in0[:,:,::rs,::rs])

LES_all = [LES_all0, LES_in0]



#--------------------------- load DNS field and prepare LES_in
# load numpy array
UVP_DNS, UVP_LES, fUVP_DNS, predictions = find_predictions(synthesis, gfilter, [dlatents, LES_all], UVP_max)

U_DNS = UVP_DNS[0,0,:,:].numpy()
V_DNS = UVP_DNS[0,1,:,:].numpy()
P_DNS = UVP_DNS[0,2,:,:].numpy()

filename = "plots_DNS.png"
print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE) #, \
            #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)
print("\nInitial residuals ------------------------:     resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))

imgA = tf.identity(UVP_DNS)
if (USE_DIFF_LES):
    fimgA = tf.identity(fUVP_DNS)
else:
    fimgA = tf.identity(UVP_LES)

nfimgA = tf.identity(LES_in0)

# set old scaling coefficients
fnUVPo, nfUVPo, fUVP_amaxo, nUVP_amaxo = find_scaling(UVP_DNS, gfilter_sub)
UVP_max = tf.concat([nUVP_amaxo, fUVP_amaxo], axis=0)

# prepare old LES_in values
if (USE_DIFF_LES):
    LES_ino = predictions[RES_LOG2-FIL-2]
    nfimgAo = tf.identity(nfimgA)

file = open("./data/BOUT.inp", 'r')
for line in file:
    if "nx =" in line:
        NX = int(line.split()[2]) - 4
    if "ny =" in line:
        NY = int(line.split()[2])
    if "nz =" in line:
        NZ = int(line.split()[2])
    if "Lx =" in line:
        LX = float(line.split()[2])
    if "Ly =" in line:
        LY = float(line.split()[2])
    if "Lz =" in line:
        LZ = float(line.split()[2])

NX = NX*RS
NZ = NZ*RS

x = np.linspace(0,LX,NX)
y = np.linspace(0,LY,NY)
z = np.linspace(0,LZ,NZ)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

print("Grid values NX,NY,NZ,LX,LY,LZ", NX,NY,NZ,LX,LY,LZ)



#------------------------------------------------------ initialize the flow
def initFlow(npv):

    if (BATCH_SIZE!=NY):
        print("=============== Abort!!! BATCH_SIZE not equal to NY dimensions")
        return(0)

    # pass delx and dely
    delx_LES = npv[0]
    dely_LES = npv[1]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    print("delx, delx_LES, N_DNS, N_LES ", delx, delx_LES, N_DNS, N_LES)

    if (RUN_DNS):

        if (DIMS_3D):
            U_LES = imgA[:,0,:,:]
            V_LES = imgA[:,1,:,:]
            P_LES = imgA[:,2,:,:]
            U_LES = tf.transpose(U_LES, perm=[1,0,2])
            V_LES = tf.transpose(V_LES, perm=[1,0,2])
            P_LES = tf.transpose(P_LES, perm=[1,0,2])
        else:
            U_LES = (imgA[0,0,:,:])
            V_LES = (imgA[0,1,:,:])
            P_LES = (imgA[0,2,:,:])
            
    else:

        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)
        print("Starting residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
            .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

        # find fields
        if (DIMS_3D):
            U_LES = fimgA[:,0,:,:]
            V_LES = fimgA[:,1,:,:]
            P_LES = fimgA[:,2,:,:]
            U_LES = tf.transpose(U_LES, perm=[1,0,2])
            V_LES = tf.transpose(V_LES, perm=[1,0,2])
            P_LES = tf.transpose(P_LES, perm=[1,0,2])
        else:        
            U_LES = fimgA[0, 0, :, :]
            V_LES = fimgA[0, 1, :, :]
            P_LES = fimgA[0, 2, :, :]

    
    # pass values back
    U_LES = tf.reshape(U_LES, [-1])
    V_LES = tf.reshape(V_LES, [-1])
    P_LES = tf.reshape(P_LES, [-1])
    
    U_LES = tf.cast(U_LES, dtype="float64")
    V_LES = tf.cast(V_LES, dtype="float64")
    P_LES = tf.cast(P_LES, dtype="float64")

    U_LES = U_LES.numpy()
    V_LES = V_LES.numpy()
    P_LES = P_LES.numpy()

    rnpv = np.concatenate((U_LES, V_LES, P_LES), axis=0)
    
    print("------------------------------------- Done StylES initialization -------------\n\n")

    return rnpv





#------------------------------------------------------ find LES terms
def findLESTerms(pLES):

    global pPrint, rLES, z0, simtimeo, pStepo
    global UVP_max, tollDNS
    global nfUVPo, fnUVPo
    global nUVP_amaxo, fUVP_amaxo
    global LES_ino, nfimgAo, UVP_DNS

        
    #--------------------------- pass values from BOUT++
    if (PROFILE_BOUT):
        tstart = time.time()

    pLES = pLES.astype(DTYPE)
    
    pStep      = int(pLES[0])
    pStepStart = int(pLES[1])
    
    delx_LES = pLES[2]
    dely_LES = delx_LES
    simtime  = pLES[3]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    if (IMPLICIT):
        BOUT_fU = pLES[4+0*SIZE:4+1*SIZE]
        BOUT_fV = pLES[4+1*SIZE:4+2*SIZE]
        BOUT_fP = pLES[4+2*SIZE:4+3*SIZE]
        BOUT_pV = pLES[4+3*SIZE:4+4*SIZE]
        BOUT_pN = pLES[4+4*SIZE:4+5*SIZE]

        fU           = np.reshape(BOUT_fU, (N_LES, BATCH_SIZE, N_LES))
        fV           = np.reshape(BOUT_fV, (N_LES, BATCH_SIZE, N_LES))
        fP           = np.reshape(BOUT_fP, (N_LES, BATCH_SIZE, N_LES))
        pPhiVort_LES = np.reshape(BOUT_pV, (N_LES, BATCH_SIZE, N_LES))
        pPhiN_LES    = np.reshape(BOUT_pN, (N_LES, BATCH_SIZE, N_LES))        
    else:
        BOUT_fU = pLES[4+0*SIZE:4+1*SIZE]
        BOUT_fV = pLES[4+1*SIZE:4+2*SIZE]
        BOUT_fP = pLES[4+2*SIZE:4+3*SIZE]
        fU = np.reshape(BOUT_fU, (N_LES, BATCH_SIZE, N_LES))
        fV = np.reshape(BOUT_fV, (N_LES, BATCH_SIZE, N_LES))
        fP = np.reshape(BOUT_fP, (N_LES, BATCH_SIZE, N_LES))
    

    #--------------------------- prepare LES field in 
    # normalize
    U_min = np.min(fU)
    U_max = np.max(fU)
    V_min = np.min(fV)
    V_max = np.max(fV)
    P_min = np.min(fP)
    P_max = np.max(fP)

    fU_amax = max(np.absolute(U_min), np.absolute(U_max))
    fV_amax = max(np.absolute(V_min), np.absolute(V_max))
    fP_amax = max(np.absolute(P_min), np.absolute(P_max))
    
    nfU = fU/fU_amax
    nfV = fV/fV_amax
    nfP = fP/fP_amax
    if (DIMS_3D):
        nfU = tf.convert_to_tensor(nfU, dtype=DTYPE)
        nfV = tf.convert_to_tensor(nfV, dtype=DTYPE)
        nfP = tf.convert_to_tensor(nfP, dtype=DTYPE)
        nfU = tf.transpose(nfU, [1,0,2])
        nfV = tf.transpose(nfV, [1,0,2])
        nfP = tf.transpose(nfP, [1,0,2])
        nfU = nfU[:,tf.newaxis,:,:]
        nfV = nfV[:,tf.newaxis,:,:]
        nfP = nfP[:,tf.newaxis,:,:]
        nfimgA = tf.concat([nfU,nfV,nfP], axis=1)
    else:
        nfimgA = np.concatenate([nfU[np.newaxis,np.newaxis, :, :], nfV[np.newaxis,np.newaxis,:,:], nfP[np.newaxis,np.newaxis,:,:]], axis=1)
        nfimgA = tf.convert_to_tensor(nfimgA, dtype=DTYPE)

    # set new LES_in
    if (USE_DIFF_LES):
        LES_diff = nfimgA-nfimgAo
        LES_in   = LES_ino + LES_diff
        LES_all  = [LES_all0, LES_in]
        LES_ino  = tf.identity(LES_in)
        nfimgAo  = tf.identity(nfimgA)
    else:    
        LES_all = [LES_all0, nfimgA]

    # find new scaling
    fnUVPo, nfUVPo, fUVP_amaxo, nUVP_amaxo, _ = find_scaling_new(UVP_DNS, fnUVPo, nfUVPo, nUVP_amaxo, fUVP_amaxo, gfilter_sub)
    UVP_max = tf.concat([nUVP_amaxo, fUVP_amaxo], axis=0)

    # end prepare phase
    if (PROFILE_BOUT):
        print("prepare ", time.time() - tstart)
        tstart = time.time()

        
    #--------------------------- find reconstructed field
    UVP_DNS = find_predictions(synthesis, gfilter, [dlatents, LES_all], UVP_max, find_fDNS=False)
    # resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)
    # print("Starting residuals: step {0:6d} simtime {1:3e} resREC {2:3e} resLES {3:3e} resDNS {4:3e} loss_fil {5:3e}" \
    #     .format(pStep, simtime, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

    
    if (PROFILE_BOUT):
        print("inference ", time.time() - tstart)
        tstart = time.time()


    #--------------------------- set global variables
    if (pStep==pStepStart):
        tollDNS  = tollLES
        pPrint   = simtime
        maxit    = lr_DNS_maxIt
        simtimeo = simtime
        delt     = 0.0
        pStepo   = pStep
    else:
        tollDNS  = tollLES
        maxit    = LES_pass
        delt     = (simtime - simtimeo)/(pStep - pStepo)
        simtimeo = simtime
        pStepo   = pStep
   
    
    #--------------------------- find poisson terms
    spacingFactor = tf.constant(1.0/(12.0*delx*dely), dtype=DTYPE)        
    F = UVP_DNS[:, 1:2, :, :]
    G = UVP_DNS[:, 2:3, :, :]
    fpPhiVort_DNS = find_bracket(F, G, gfilter_single, spacingFactor)
    G = UVP_DNS[:, 0:1, :, :]
    fpPhiN_DNS = find_bracket(F, G, gfilter_single, spacingFactor)

    
    if (IMPLICIT):
        tauPhiVort = - (fpPhiVort_DNS - pPhiVort_LES) # find sub-grid scale model and use in the implicit diffusion
        tauPhiN    = - (fpPhiN_DNS    - pPhiN_LES)
    else:
        tauPhiVort = fpPhiVort_DNS  # fully explicit
        tauPhiN    = fpPhiN_DNS

    # convective explicit and sub-grid scale model implicit

    if (PROFILE_BOUT):
        print("bracket ", time.time() - tstart)
        tstart = time.time()


    #--------------------------- pass back diffusion terms
    tauPhiVort = tf.reshape(tauPhiVort, [-1])
    tauPhiN    = tf.reshape(tauPhiN,    [-1])
    
    tauPhiVort = tf.cast(tauPhiVort, dtype="float64")
    tauPhiN    = tf.cast(tauPhiN, dtype="float64")

    tauPhiVort = tauPhiVort.numpy()
    tauPhiN    = tauPhiN.numpy()

    LES_it = np.asarray([0], dtype="float64")
    rLES = np.concatenate((LES_it, tauPhiVort, tauPhiN), axis=0)

    if (PROFILE_BOUT):
        print("concatenate ", time.time() - tstart)
        tstart = time.time()


    #--------------------------- save field values
    if (simtime>=pPrint):
        pPrint = pPrint + pPrintFreq

        # find DNS fields
        U_DNS = UVP_DNS[:,0,:,:].numpy()
        V_DNS = UVP_DNS[:,1,:,:].numpy()
        P_DNS = UVP_DNS[:,2,:,:].numpy()

        # transpose 
        U_DNS = np.transpose(U_DNS, (1,0,2))
        V_DNS = np.transpose(V_DNS, (1,0,2))
        P_DNS = np.transpose(P_DNS, (1,0,2))
        U_DNS = np.ascontiguousarray(U_DNS)
        V_DNS = np.ascontiguousarray(V_DNS)
        P_DNS = np.ascontiguousarray(P_DNS)

        # save as vts
        filename = "./results_StylES/fields/fields_DNS_" + str(pStep).zfill(7)
        gridToVTK(filename, X, Y, Z, pointData={"n": U_DNS, "phi": V_DNS, "vort": P_DNS})
    

    if (PROFILE_BOUT):
        print("saving ", time.time() - tstart)


    return rLES




def writePoissonDNS(pLES):

    global pPrint, rLES, z0

    # pass values from BOUT++
    pLES = pLES.astype(DTYPE)
    
    pStep      = int(pLES[0])
    pStepStart = int(pLES[1])

    delx_LES = pLES[2]
    dely_LES = delx_LES
    simtime  = pLES[3]
    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
 
    # print("L, delx and delx_LES are: ", L, delx, delx_LES, N_DNS, N_LES)

    BOUT_U_LES = pLES[4+0*N_DNS*N_DNS:4+1*N_DNS*N_DNS]
    BOUT_V_LES = pLES[4+1*N_DNS*N_DNS:4+2*N_DNS*N_DNS]
    BOUT_P_LES = pLES[4+2*N_DNS*N_DNS:4+3*N_DNS*N_DNS]
    BOUT_N_LES = pLES[4+3*N_DNS*N_DNS:4+4*N_DNS*N_DNS]
    BOUT_F_LES = pLES[4+4*N_DNS*N_DNS:4+5*N_DNS*N_DNS]        

    U_LES = np.reshape(BOUT_U_LES, (N_DNS, N_DNS))
    V_LES = np.reshape(BOUT_V_LES, (N_DNS, N_DNS))
    P_LES = np.reshape(BOUT_P_LES, (N_DNS, N_DNS))
    fpPhiVort_DNS = np.reshape(BOUT_N_LES, (N_DNS, N_DNS))
    fpPhiN_DNS    = np.reshape(BOUT_F_LES, (N_DNS, N_DNS))

    filename = "./results_StylES/fields/fields_PoissonDNS_" + str(pStep).zfill(7)
    np.savez(filename, pStep=pStep, simtime=simtime, U=fpPhiVort_DNS, V=fpPhiN_DNS, P=fpPhiN_DNS)

    return fpPhiVort_DNS



# test
if (RUN_TEST):
    dummy=np.array([0.2, 0.2])
    initFlow(dummy)





# #------------------------------------------- Extra pieces....


    # fpPhiVort_DNS = tf.zeros([N_LES,N_LES], dtype=DTYPE)
    # fpPhiN_DNS    = tf.zeros([N_LES,N_LES], dtype=DTYPE)

    # fpPhiVort_DNS = tf.reshape(fpPhiVort_DNS, [-1])
    # fpPhiN_DNS    = tf.reshape(fpPhiN_DNS,    [-1])
    
    # fpPhiVort_DNS = tf.cast(fpPhiVort_DNS, dtype="float64")
    # fpPhiN_DNS    = tf.cast(fpPhiN_DNS, dtype="float64")

    # fpPhiVort_DNS = fpPhiVort_DNS.numpy()
    # fpPhiN_DNS    = fpPhiN_DNS.numpy()

    # LES_it = np.asarray([0], dtype="float64")
    # rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS), axis=0)     


    # return rLES
