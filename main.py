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
import PIL
import time
import sys
import pathlib
import datetime

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *



#-------------------------------------prepare for run
# clean folders
os.system("rm -rf logs/*")
if os.path.isdir("images"):
    os.system("rm -rf images/*")
else:    
    os.system("mkdir images")

current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_train_log        = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(dir_train_log)

if (PROFILE):
    tf.summary.trace_on(graph=True, profiler=True)


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



#-------------------------------------define data augmentation
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
# ])



#-------------------------------------define dataset
list_ds = tf.data.Dataset.list_files(str(DATASET + '*.png' ))

def decode_img(img):
  #convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  if (NUM_CHANNELS==1):
    img = tf.image.rgb_to_grayscale(img)
  
  #Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  
  #resize the image to the desired size.
  img_out = []
  for res in range(2, RES_LOG2 + 1):
    r_img = tf.image.resize(img, [2**res, 2**res])
    if (NUM_CHANNELS>1):
        r_img = tf.transpose(r_img)
    img_out.append(r_img)
    
  return img_out


def process_path(file_path):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)



#-------------------------------------prepare training dataset
def prepare_for_training(ds, cache=True, batch_size=0, shuffle_buffer_size=BUFFER_SIZE, augment=True):
    #This is a small dataset, only load it once, and keep it in memory.
    #use `.cache(filename)` to cache preprocessing work for datasets that don't
    #fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    #Repeat forever
    ds = ds.repeat()

    #take batch size
    ds = ds.batch(batch_size)

    # if augment:
    #     ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    #`prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds



#-------------------------------------main: train the model
def main():
    train_images = prepare_for_training(labeled_ds, batch_size=BATCH_SIZE)
    train(train_images, GEN_LR, DIS_LR, train_summary_writer)


if __name__ == "__main__":
    main()
