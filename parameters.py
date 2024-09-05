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
import tensorflow as tf
import numpy as np
import os


# General parameters
DTYPE = "float64"        # Data type to use for activations and outputs.
if (DTYPE=="float64"):
    SMALL = 1.0e-8
    tf.keras.backend.set_floatx('float64')
else:
    SMALL = 1.0e-15

DEBUG = False
MINVALRAN = -1.0
MAXVALRAN =  1.0
if (DEBUG):
    tf.debugging.experimental.enable_dump_debug_info("./debug_logdir", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    tf.get_logger().setLevel("ALL")           # DEBUG = all messages are logged (default behavior)
else:                                         # INFO = INFO messages are not printed 
    tf.get_logger().setLevel("ERROR")

SEED = 0
SEED_RESTART = 5

tf.random.set_seed(seed=SEED)  # ideally this should be set on if DEBUG is true...


TESTCASE          = 'HW' 
DATASET           = '../../data/BOUT_runs/HW_3D/HW_N512x16x512_perX/fields_npz/'
CHKP_DIR          = './checkpoints/'
CHKP_PREFIX       = os.path.join(CHKP_DIR, 'ckpt')
PROFILE           = False
CONVERTTOTFRECORD = False
USE_GPU           = True
DEVICE_TYPE       = 'GPU'
AUTOTUNE          = tf.data.experimental.AUTOTUNE
READ_NUMPY_ARRAYS = True
SAVE_NUMPY_ARRAYS = False

if DEVICE_TYPE in ('CPU', 'IPU'):
    data_format = 'NHWC'
    TRANSPOSE_FOR_CONV2D = [0,2,3,1]
    TRANSPOSE_FROM_CONV2D= [0,3,1,2]
elif DEVICE_TYPE == 'GPU':
    data_format = 'NCHW'
    TRANSPOSE_FOR_CONV2D = [0,1,2,3]
    TRANSPOSE_FROM_CONV2D = [0,1,2,3]

# Network hyper-parameters
OUTPUT_DIM        = 512
BATCH_SIZE        = 16  # remember this shoudl NOT be bigger than dataset length!
DPI               = 100*max(1,int(OUTPUT_DIM/256))
LATENT_SIZE       = 512            # Size of the lantent space, which is constant in all mapping layers 
GM_LRMUL          = 0.01           # Learning rate multiplier
BLUR_FILTER       = [1, 2, 1, ]    # Low-pass filter to apply when resampling activations. None = no filtering.
GAIN              = np.sqrt(2.0)
FMAP_BASE         = 8192    # Overall multiplier for the number of feature maps.
FMAP_DECAY        = 1.0     # log2 feature map reduction when doubling the resolution.
FMAP_MAX          = 512     # Maximum number of feature maps in any layer.
RES_LOG2          = int(np.log2(OUTPUT_DIM))
FIL               = 3 # number of layers below the DNS  
IFIL              = FIL-1  # number of layers below the DNS  
G_LAYERS          = RES_LOG2*2 - 2  # Numer of layers  
G_LAYERS_FIL      = (RES_LOG2-FIL)*2 - 2   # Numer of layers for the filter
M_LAYERS          = 2*(RES_LOG2 - FIL) - 2  # end of medium layers (ideally equal to the filter...)
C_LAYERS          = 2  # end of coarse layers 
NUM_CHANNELS      = 3      # Number of input color channels. Overridden based on dataset.
SCALING_UP        = tf.math.exp( tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
SCALING_DOWN      = tf.math.exp(-tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
R1_GAMMA          = 10  # Gradient penalty coefficient
BUFFER_SIZE       = 5000 #same size of the number of images in DATASET
AMP_NOISE_MAX     = 1.0
NC_NOISE          = 50
NC2_NOISE         = int(NC_NOISE/2)
USE_VORTICITY     = True
USE_IMGSLES       = True
RANDOMIZE_NOISE   = False

# Training hyper-parameters
TOT_ITERATIONS = 500000
PRINT_EVERY    = 100
IMAGES_EVERY   = 1000
SAVE_EVERY     = 10000
IRESTART       = False

# learning rates
LR_GEN           = 7.5e-4
DECAY_STEPS_GEN  = TOT_ITERATIONS+1
DECAY_RATE_GEN   = 1.0
STAIRCASE_GEN    = True
BETA1_GEN        = 0.0
BETA2_GEN        = 0.99

LR_FIL           = 1.0e-4
DECAY_STEPS_FIL  = 100000
DECAY_RATE_FIL   = 0.1
STAIRCASE_FIL    = True
BETA1_FIL        = 0.0
BETA2_FIL        = 0.99

LR_DIS           = 3.0e-3
DECAY_STEPS_DIS  = TOT_ITERATIONS+1
DECAY_RATE_DIS   = 1.0
STAIRCASE_DIS    = True
BETA1_DIS        = 0.0
BETA2_DIS        = 0.99


# Reconstruction parameters
N_DNS           = int(2**RES_LOG2)
N_LES           = int(2**(RES_LOG2-FIL))
N_DNS2          = int(N_DNS/2)
N_LES2          = int(N_LES/2)
NY2             = max(1,int(BATCH_SIZE/2))
RS              = int(2**FIL)
RS2             = int(RS/2)
N2L             = N_LES2-RS2
N2R             = N_LES2+RS2+1
LEN_DOMAIN      = 50.176  # for 2D HWLEN_DOMAIN
DELX            = LEN_DOMAIN/N_DNS
DELY            = LEN_DOMAIN/N_DNS
DELX_LES        = LEN_DOMAIN/N_LES
DELY_LES        = LEN_DOMAIN/N_LES
INIT_SCA        = 2.0  # 5 10, 15
NC_NOISE_IN     = 1000
NC2_NOISE_IN    = int(NC_NOISE_IN/2)
GAUSSIAN_FILTER = True
FILE_DNS_N256    = "../../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N256/fields/fields_run0_time501.npz"
FILE_DNS_N512    = "../../../data/BOUT_runs/HW_3D/HW_N512x16x512_perX/fields_npz/fields_run0_time298.npz"
FILE_DNS_N1024   = "../../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N1024/fields/fields_run0_time440.npz"
FILE_DNS_N256_3D = "../../../data/BOUT_runs/HW_3D/HW_larger/HW_N256/fields_run6_time400.npz"
FILE_DNS_N512_3D = "../../../data/BOUT_runs/HW_3D/HW_larger/HW_N256/fields_run1_time300.npz"


# learning rate for latent space optimizer
lr_DNS_maxIt  = 100000
lr_DNS_POLICY = "EXPONENTIAL"   # "EXPONENTIAL" or "PIECEWISE"
lr_DNS_STAIR  = False
lr_DNS        = 1.0e-3   # exponential policy initial learning rate
lr_DNS_RATE   = 1.0       # exponential policy decay rate
lr_DNS_STEP   = lr_DNS_maxIt     # exponential policy decay step
lr_DNS_EXP_ST = False      # exponential policy staircase
lr_DNS_BOUNDS = [100, 200, 300]             # piecewise policy bounds
lr_DNS_VALUES = [100.0, 50.0, 20.0, 10.0]   # piecewise policy values
lr_DNS_BETA1  = 0.0
lr_DNS_BETA2  = 0.99

