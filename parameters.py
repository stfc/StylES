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
DTYPE = "float32"        # Data type to use for activations and outputs.
if (DTYPE=="float64"):
    SMALL = 1.0e-8
    tf.keras.backend.set_floatx('float64')
else:
    SMALL = 1.0e-15

DEBUG = False
tf.random.set_seed(seed=0)

if (DEBUG):
    MINVALRAN = 0.5
    MAXVALRAN = 0.5
else:
    MINVALRAN = 0.0
    MAXVALRAN = None

TESTCASE          = '2D-HIT' 
DATASET           = './LES_Solvers/fields/'
CHKP_DIR          = './checkpoints/'
CHKP_PREFIX       = os.path.join(CHKP_DIR, 'ckpt')
PROFILE           = False
CONVERTTOTFRECORD = False
USE_GPU           = True
AUTOTUNE          = tf.data.experimental.AUTOTUNE
READ_NUMPY_ARRAYS = True
SAVE_NUMPY_ARRAYS = False

# Network hyper-parameters
OUTPUT_DIM        = 256
LATENT_SIZE       = 512            # Size of the lantent space, which is constant in all mapping layers 
GM_LRMUL          = 0.01           # Learning rate multiplier
BLUR_FILTER       = [1, 2, 1, ]    # Low-pass filter to apply when resampling activations. None = no filtering.
GAIN              = np.sqrt(2.0)
FMAP_BASE         = 8192    # Overall multiplier for the number of feature maps.
FMAP_DECAY        = 1.0     # log2 feature map reduction when doubling the resolution.
FMAP_MAX          = 512     # Maximum number of feature maps in any layer.
RES_LOG2          = int(np.log2(OUTPUT_DIM))
RES_LOG2_FIL      = RES_LOG2-3    # fix filter layer
NUM_CHANNELS      = 3                # Number of input color channels. Overridden based on dataset.
G_LAYERS          = RES_LOG2*2 - 2  # Numer of layers  
G_LAYERS_FIL      = RES_LOG2_FIL*2 - 2   # Numer of layers for the filter
SCALING_UP        = tf.math.exp( tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
SCALING_DOWN      = tf.math.exp(-tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
R1_GAMMA          = 10  # Gradient penalty coefficient
BUFFER_SIZE       = 5000 #same size of the number of images in DATASET
NEXAMPLES         = 1
RANDOMIZE_NOISE   = True

# Training hyper-parameters
TOT_ITERATIONS = 200000
PRINT_EVERY    = 1000
IMAGES_EVERY   = 1000
SAVE_EVERY     = 100000
BATCH_SIZE     = NEXAMPLES
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

