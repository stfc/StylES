#----------------------------------------------------------------------------------------------
#
#    Copyright: STFC - Hartree Centre (2021)
#
#    Author: Jony Castagna
#
#    Licence: most of this material is taken from StyleGAN and MSG-StyleGAN. Please use same
#             licence policy
#
#-----------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import os


# General parameters
DTYPE = "float32"        # Data type to use for activations and outputs.
if (DTYPE=="float64"):
    tf.keras.backend.set_floatx('float64')
TRAIN             = True
DATASET           = './LES_Solvers/uvw/'
CHKP_DIR          = './checkpoints/'
CHKP_PREFIX       = os.path.join(CHKP_DIR, 'ckpt')
PROFILE           = False
CONVERTTOTFRECORD = False
USE_GPU           = True
AUTOTUNE          = tf.data.experimental.AUTOTUNE
READ_NUMPY_ARRAYS = False
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
NUM_CHANNELS      = 3                # Number of input color channels. Overridden based on dataset.
G_LAYERS          = RES_LOG2*2 - 2  # Numer of layers  
SCALING_UP        = tf.math.exp( tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
SCALING_DOWN      = tf.math.exp(-tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
R1_GAMMA          = 10  # Gradient penalty coefficient
BUFFER_SIZE       = 1000 #same size of the number of images in DATASET
NEXAMPLES         = 4


# Training hyper-parameters
TOT_ITERATIONS = 25000
PRINT_EVERY    = 1000
IMAGES_EVERY   = 2500
SAVE_EVERY     = 25000
BATCH_SIZE     = NEXAMPLES
IRESTART       = False
LR             = 3.0e-3
DECAY_STEPS    = TOT_ITERATIONS
DECAY_RATE     = 1.0
STAIRCASE      = False
G_SMOOTH       = 10.0
if G_SMOOTH > 0.0:
    Gs_beta = 0.5**tf.math.divide(tf.cast(BATCH_SIZE, DTYPE), G_SMOOTH * 1000.0)
else:
    Gs_beta = 0.0