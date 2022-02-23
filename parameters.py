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

DEBUG = False
if (DEBUG):
    tf.random.set_seed(seed=0)
    MINVALRAN = 0.5
    MAXVALRAN = 0.5
else:
    MINVALRAN = 0.0
    MAXVALRAN = None


TRAIN             = False
DATASET           = './LES_Solvers/temp/'
CHKP_DIR          = './checkpoints/'
CHKP_PREFIX       = os.path.join(CHKP_DIR, 'ckpt')
PROFILE           = False
CONVERTTOTFRECORD = False
USE_GPU           = True
AUTOTUNE          = tf.data.experimental.AUTOTUNE
READ_NUMPY_ARRAYS = True
SAVE_NUMPY_ARRAYS = False

# Network hyper-parameters
OUTPUT_DIM        = 1024
LATENT_SIZE       = 512            # Size of the lantent space, which is constant in all mapping layers 
GM_LRMUL          = 0.01           # Learning rate multiplier
BLUR_FILTER       = [1, 2, 1, ]    # Low-pass filter to apply when resampling activations. None = no filtering.
GAIN              = np.sqrt(2.0)
FMAP_BASE         = 8192    # Overall multiplier for the number of feature maps.
FMAP_DECAY        = 1.0     # log2 feature map reduction when doubling the resolution.
FMAP_MAX          = 512     # Maximum number of feature maps in any layer.
RES_LOG2          = int(np.log2(OUTPUT_DIM))
RES_LOG2_FIL      = RES_LOG2-3    # fix filter layer
RES_TARGET        = 4   # 4=16x16
NUM_CHANNELS      = 3                # Number of input color channels. Overridden based on dataset.
G_LAYERS          = RES_LOG2*2 - 2  # Numer of layers  
G_LAYERS_FIL      = RES_LOG2_FIL*2 - 2   # Numer of layers for the filter
SCALING_UP        = tf.math.exp( tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
SCALING_DOWN      = tf.math.exp(-tf.cast(64.0, DTYPE) * tf.cast(tf.math.log(2.0), DTYPE))
R1_GAMMA          = 10  # Gradient penalty coefficient
BUFFER_SIZE       = 1000 #same size of the number of images in DATASET
NEXAMPLES         = 1


# Training hyper-parameters
TOT_ITERATIONS = 50000
PRINT_EVERY    = 100
IMAGES_EVERY   = 1000
SAVE_EVERY     = 50000 
BATCH_SIZE     = NEXAMPLES
IRESTART       = False
LR             = 3.0e-3
DECAY_STEPS    = TOT_ITERATIONS
DECAY_RATE     = 1.0
STAIRCASE      = True

