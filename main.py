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
import scipy as sc

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *



#-------------------------------------prepare for run
# clean folders
os.system("rm -rf logs/*")
os.system("rm -rf images/*")
for res in range(2,RES_LOG2+1):
    cmd="mkdir -p images/image_{:d}x{:d}".format(2**res,2**res)
    os.system(cmd)

current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_train_log        = 'logs/' + current_time + '/train'
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
if (READ_NUMPY):
    list_ds = tf.data.Dataset.list_files(str(DATASET + '*.npz' ))
else:
    list_ds = tf.data.Dataset.list_files(str(DATASET + '*.png' ))

def decode_img(img):
    #convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    if (NUM_CHANNELS==1):
        img = tf.image.rgb_to_grayscale(img)
  
    #Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, DTYPE)
  
    #resize the image to the desired size.
    img_out = []
    for res in range(2, RES_LOG2 + 1):
        r_img = tf.image.resize(img, [2**res, 2**res])
        r_img = tf.transpose(r_img)
        img_out.append(r_img)
    
    return img_out


def process_path(file_path):
    if (READ_NUMPY):
        U = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
        P = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
        V = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
        U, V, P = load_fields(DATASET + "restart_N32.npz")
        U = np.cast[DTYPE](U)
        V = np.cast[DTYPE](V)
        P = np.cast[DTYPE](P)
        img = []
        N = OUTPUT_DIM
        for res in range(RES_LOG2-1):
            pow2 = 2**(res+2)
            data = np.zeros([3,pow2,pow2], dtype=DTYPE)
            s = pow2/N
            data[0,:,:] = sc.ndimage.interpolation.zoom(U, s, order=3, mode='wrap')
            data[1,:,:] = sc.ndimage.interpolation.zoom(V, s, order=3, mode='wrap')
            data[2,:,:] = sc.ndimage.interpolation.zoom(P, s, order=3, mode='wrap')
            img.append(data)
    else:
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

    if (TRAIN):

        train_images = prepare_for_training(labeled_ds, batch_size=BATCH_SIZE)
        train(train_images, GEN_LR, DIS_LR, train_summary_writer)

    else:

        pass
        # # load NN and find convolutional layers used in the upsampling
        # checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))
        # conv_layers = []
        # for layer in synthesis.layers:
        #     if ("up_conv" in layer.name):
        #         conv_layers.append(layer)

        # # generate image
        # tf.random.set_seed(1)
        # input = tf.random.uniform([BATCH_SIZE, LATENT_SIZE])
        # dlatents    = mapping_ave(input, training=False)
        # predictions = synthesis_ave(dlatents, training=False)

        # # find non linear terms
        # U = predictions[RES_LOG2-2][0,0,:,:]
        # V = predictions[RES_LOG2-2][0,1,:,:]
        # UU = U*U
        # VV = V*V

        # # find filtered non linear terms
        # tlen = len(conv_layers)
        # for i in range(tlen-1, -1, -1):
        #     UU = conv_layers[i](UU)
        #     VV = conv_layers[i](VV)

        # Ufil = UU.asnumpy()
        # Vfil = VV.asnumpy()

        # Ufil = (Ufil - np.min(UU))/(np.max(UU) - np.min(UU))
        # Vfil = (Vfil - np.min(VV))/(np.max(VV) - np.min(VV))

        # img = Image.fromarray(np.uint8(Ufil*255), 'RGB')
        # filename = "Ufiltered.png"
        # size = 32, 32
        # img.thumbnail(size)
        # img.save(filename)      




if __name__ == "__main__":
    main()
