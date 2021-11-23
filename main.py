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
import datetime

from PIL import Image

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *
from IO_functions import *



#------------------------------------- prepare for run
# clean folders
os.system("rm -rf logs/*")
os.system("rm -rf images/*")
for reslog in range(2,RES_LOG2+1):
    cmd="mkdir -p images/image_{:d}x{:d}".format(2**reslog,2**reslog)
    os.system(cmd)

current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_train_log        = 'logs/train'
train_summary_writer = tf.summary.create_file_writer(dir_train_log)

# enable profiler
if (PROFILE):
    tf.summary.trace_on(graph=True, profiler=True)

# use GPU by default
if (not USE_GPU):
    try:
        # Disable first GPU
        tf.config.set_visible_devices(physical_devices[1:], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for first GPU
        assert len(logical_devices) == len(physical_devices) - 1
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass




#------------------------------------- main: train the model
def main():

    train_images = prepare_for_training(labeled_ds, batch_size=BATCH_SIZE)
    train(train_images, LR, train_summary_writer)

if __name__ == "__main__":
    main()
