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
import scipy as sc

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *

from PIL import Image




#-------------------------------------------- dataset

# variable used during data loading
U = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
V = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
P = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)


# define data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=1.0, fill_mode='wrap', interpolation='bilinear'),
])



# define dataset
if (READ_NUMPY_ARRAYS):
    list_ds = tf.data.Dataset.list_files(str(DATASET + '*.npz' ))
else:
    list_ds = tf.data.Dataset.list_files(str(DATASET + '*.png' ))



# functions for processing and decoding images
def decode_img(img):
    #convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    if (NUM_CHANNELS==1):
        img = tf.image.rgb_to_grayscale(img)
  
    #Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, DTYPE)
    img = (2.0*img - 1.0)   # and then [-1,1] range
  
    #resize the image to the desired size.
    img_out = []
    for reslog in range(2, RES_LOG2 + 1):
        r_img = tf.image.resize(img, [2**reslog, 2**reslog])
        r_img = tf.transpose(r_img)
        img_out.append(r_img)
    
    return img_out



def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img



# functions for processing and decoding numpy arrays
def StyleGAN_load_fields(file_path):
    data_org = np.load(file_path)
    U_DNS_org = np.cast[DTYPE](data_org['U'])
    V_DNS_org = np.cast[DTYPE](data_org['V'])
    P_DNS_org = np.cast[DTYPE](data_org['P'])

    rs = 2
    img_out = []
    for reslog in range(RES_LOG2, 1, -1):
        res = 2**reslog
        data = np.zeros([3, res, res], dtype=DTYPE)
        if (reslog==RES_LOG2):
            fU_DNS = U_DNS_org
            fV_DNS = V_DNS_org
            fP_DNS = P_DNS_org
        else:
            if (TESTCASE=='mHW'):
                fU_DNS = sc.ndimage.gaussian_filter(fU_DNS, rs, mode=['constant','wrap'])
                fV_DNS = sc.ndimage.gaussian_filter(fV_DNS, rs, mode=['constant','wrap'])
                fP_DNS = sc.ndimage.gaussian_filter(fP_DNS, rs, mode=['constant','wrap'])
            else:
                fU_DNS = sc.ndimage.gaussian_filter(fU_DNS, rs, mode='wrap')
                fV_DNS = sc.ndimage.gaussian_filter(fV_DNS, rs, mode='wrap')
                fP_DNS = sc.ndimage.gaussian_filter(fP_DNS, rs, mode='wrap')

            fU_DNS = fU_DNS[::rs,::rs]
            fV_DNS = fV_DNS[::rs,::rs]
            fP_DNS = fP_DNS[::rs,::rs]

        # normalize the data
        minU = np.min(fU_DNS)
        maxU = np.max(fU_DNS)
        amaxU = max(abs(minU), abs(maxU))
        if (amaxU<SMALL):
            print("-----------Attention: invalid field!!!")
            exit(0)
        else:
            data[0,:,:] = fU_DNS / amaxU
        
        minV = np.min(fV_DNS)
        maxV = np.max(fV_DNS)
        amaxV = max(abs(minV), abs(maxV))
        if (amaxV<SMALL):
            print("-----------Attention: invalid field!!!")
            exit(0)
        else:
            data[1,:,:] = fV_DNS / amaxV

        minP = np.min(fP_DNS)
        maxP = np.max(fP_DNS)
        amaxP = max(abs(minP), abs(maxP))
        if (amaxP<SMALL):
            print("-----------Attention: invalid field!!!")
            exit(0)
        else:
            data[2,:,:] = fP_DNS / amaxP


        img_out = [data] + img_out

    return img_out



def process_path_numpy_arrays(path):
    list = []
    for i in range(RES_LOG2-1):
        list.append(DTYPE)
    img = tf.numpy_function(StyleGAN_load_fields, [path], list)
    return img



# define dataset. Set num_parallel_calls so multiple images are loaded/processed in parallel.
if (READ_NUMPY_ARRAYS):
    labeled_ds = list_ds.map(process_path_numpy_arrays, num_parallel_calls=AUTOTUNE)
else:
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)



# prepare training dataset
def prepare_for_training(ds, cache=True, batch_size=BATCH_SIZE, shuffle_buffer_size=BUFFER_SIZE, augment=False):

    # take batch size
    ds = ds.batch(batch_size)

    # this is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    # repeat shuffle
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # repeat forever
    ds = ds.repeat()

    # augment
    if augment:
        ds = ds.map(lambda x: (data_augmentation(x, training=True)), num_parallel_calls=AUTOTUNE)

    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds



#------------------------------ functions for visualization
def StyleGAN_save_fields(it, U, V, P):
    #filename = "fields.npz"
    filename = 'images/image_{:d}x{:d}/fields.npz'.format(OUTPUT_DIM,OUTPUT_DIM)
    #filename = 'images/image_{:d}x{:d}/fields_it_{:06d}.npz'.format(OUTPUT_DIM,OUTPUT_DIM,it)
    np.savez(filename, U=U, V=V, P=P)


def convert_to_pil_image(image, drange=[0, 1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]  # grayscale CHW => HW
        else:
            image = tf.transpose(image)  # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0, 255])
    image = tf.math.rint(image)
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.cast[DTYPE](drange_out[1]) - np.cast[DTYPE](drange_out[0])) / (
            np.cast[DTYPE](drange_in[1]) - np.cast[DTYPE](drange_in[0])
        )
        bias = np.cast[DTYPE](drange_out[0]) - np.cast[DTYPE](drange_in[0]) * scale
        data = data * scale + bias
    return data    


def check_divergence(img, res):

    # initialize arrays
    U = img[0,:,:]
    V = img[1,:,:]
    P = img[2,:,:]
    iNN  = 1.0e0/(OUTPUT_DIM*OUTPUT_DIM)

    delt = 1.0e-4   # this value is from the input file (testcases/HIT_2D)
    rho  = 1.0e0    # this value is from the input file (testcases/HIT_2D)
    L    = 0.94049  # this value is from the input file (testcases/HIT_2D)
    nu   = 1.87e-4  # this value is from the input file (testcases/HIT_2D)

    dl = L/res
    A = dl
    Dc = nu/dl*A

    # find Rhie-Chow interpolation (PWIM)
    Ue = 0.5e0*(U + nr(U, 1, 0))
    Vn = 0.5e0*(V + nr(V, 1, 0))

    Fw = A*rho*nr(Ue, -1, 0)
    Fe = A*rho*Ue
    Fs = A*rho*nr(Vn, 0, -1)
    Fn = A*rho*Vn

    Aw = Dc + 0.5e0*(np.abs(Fw) + Fw)
    Ae = Dc + 0.5e0*(np.abs(Fe) - Fe)
    As = Dc + 0.5e0*(np.abs(Fs) + Fs)
    An = Dc + 0.5e0*(np.abs(Fn) - Fn)
    Ao = rho*A*dl/delt

    Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
    iApM = 1.e0/Ap

    deltpX1 = 0.5e0*(nr(P, 1, 0) - nr(P, -1, 0))
    deltpX2 = 0.5e0*(nr(P, 2, 0) - P)    
    deltpX3 = (P - nr(P,  1, 0))

    deltpY1 = 0.5e0*(nr(P, 0, 1) - nr(P, 0, -1))
    deltpY2 = 0.5e0*(nr(P, 0, 2) - P)
    deltpY3 = (P - nr(P, 0,  1))

    Ue = 0.5e0*(nr(U, 1, 0) + U)                 \
        + 0.5e0*deltpX1*iApM*A                   \
        + 0.5e0*deltpX2*nr(iApM, 1, 0)*A         \
        + 0.5e0*deltpX3*(nr(iApM, 1, 0) + iApM)*A

    Vn = 0.5e0*(nr(V, 0, 1) + V)                 \
        + 0.5e0*deltpY1*iApM*A                   \
        + 0.5e0*deltpY2*nr(iApM, 0, 1)*A         \
        + 0.5e0*deltpY3*(nr(iApM, 0, 1) + iApM)*A

    # check divergence
    div = rho*A*np.sum(np.abs(nr(Ue, -1, 0) - Ue + nr(Vn, 0, -1) - Vn))
    div = div*iNN

    # find dU/dt term
    sU = - 0.5e0*(nr(P, 1, 0) - nr(P, -1, 0))*A
    dUdt = np.sum(np.abs(sU + Aw*nr(U, -1, 0) + Ae*nr(U, 1, 0) + As*nr(U, 0, -1) + An*nr(U, 0, 1)))
    dUdt = dUdt*iNN

    # find dV/dt term
    sV = - 0.5e0*(nr(P, 0, 1) - nr(P, 0, -1))*A
    dVdt = np.sum(np.abs(sV + Aw*nr(V, -1, 0) + Ae*nr(V, 1, 0) + As*nr(V, 0, -1) + An*nr(V, 0, 1)))
    dVdt = dVdt*iNN

    return div, dUdt, dVdt


def check_divergence_staggered(img, res):
    
    # initialize arrays
    U = img[0,:,:]
    V = img[1,:,:]
    P = img[2,:,:]
    iNN  = 1.0e0/(OUTPUT_DIM*OUTPUT_DIM)

    delt = 1.0e-4   # this value is from the input file (testcases/HIT_2D)
    rho  = 1.0e0    # this value is from the input file (testcases/HIT_2D)
    L    = 0.94049  # this value is from the input file (testcases/HIT_2D)
    nu   = 1.87e-4  # this value is from the input file (testcases/HIT_2D)

    dl = L/res
    A = dl
    Dc = nu/dl*A

    # check divergence
    div = rho*A*np.sum(np.abs(nr(U, 1, 0) - U + nr(V, 0, 1) - V))
    div = div*iNN

    # x-direction
    Fw = A*rho*0.5e0*(U            + nr(U, -1, 0))
    Fe = A*rho*0.5e0*(nr(U,  1, 0) + U           )
    Fs = A*rho*0.5e0*(V            + nr(V, -1, 0))
    Fn = A*rho*0.5e0*(nr(V,  0, 1) + nr(V, -1, 1))

    Aw = Dc + 0.5e0*Fw #0.5e0*(np.abs(Fw) + Fw)
    Ae = Dc - 0.5e0*Fe #0.5e0*(np.abs(Fe) - Fe)
    As = Dc + 0.5e0*Fs #0.5e0*(np.abs(Fs) + Fs)
    An = Dc - 0.5e0*Fn #0.5e0*(np.abs(Fn) - Fn)
    dUdt = np.sum(np.abs(Aw*nr(U, -1, 0) + Ae*nr(U, 1, 0) + As*nr(U, 0, -1) + An*nr(U, 0, 1) - (P - nr(P, -1, 0))*A))
    dUdt = dUdt*iNN

    # y-direction
    Fw = A*rho*0.5e0*(U             + nr(U, 0, -1))
    Fe = A*rho*0.5e0*(nr(U,  1,  0) + nr(U, 1, -1))
    Fs = A*rho*0.5e0*(nr(V,  0, -1) + V           )
    Fn = A*rho*0.5e0*(V             + nr(V, 0,  1))

    Aw = Dc + 0.5e0*Fw #0.5e0*(np.abs(Fw) + Fw)
    Ae = Dc - 0.5e0*Fe #0.5e0*(np.abs(Fe) - Fe)
    As = Dc + 0.5e0*Fs #0.5e0*(np.abs(Fs) + Fs)
    An = Dc - 0.5e0*Fn #0.5e0*(np.abs(Fn) - Fn)
    dVdt = np.sum(np.abs(Aw*nr(V, -1, 0) + Ae*nr(V, 1, 0) + As*nr(V, 0, -1) + An*nr(V, 0, 1) - (P - nr(P, 0, -1))*A))
    dVdt = dVdt*iNN

    return div, dUdt, dVdt


def generate_and_save_images(mapping, synthesis, input, iteration):
    dlatents    = mapping(input[0], training=False)
    predictions = pre_synthesis(dlatents, training=False)
    new_predictions = [predictions[0:RES_LOG2-FIL-2], input[1]]
    predictions = synthesis([dlatents, new_predictions], training=False)

    div  = np.zeros(RES_LOG2-1)
    momU = np.zeros(RES_LOG2-1)
    momV = np.zeros(RES_LOG2-1)

    colors = plt.cm.jet(np.linspace(0,1,4))
    lineColor = colors[0]

    for reslog in range(RES_LOG2-1):
        res = 2**(reslog+2)

        # setup figure size
        nr = NEXAMPLES
        nc = 3
        fig, axs = plt.subplots(nr,nc, figsize=(2.5*nc, 2.5), dpi=DPI)
        plt.subplots_adjust(wspace=0, hspace=0)
        axs = axs.ravel()
        img = predictions[reslog]

        # save the highest dimension and first image of the batch as numpy array
        if (res==OUTPUT_DIM and SAVE_NUMPY_ARRAYS):
            StyleGAN_save_fields(iteration, img[0,0,:,:], img[0,1,:,:], img[0,2,:,:])

        for i in range(NEXAMPLES):

            if (NUM_CHANNELS == 3):

                # print divergence
                if (TESTCASE=='HIT_2D'):
                    divergence, dUdt, dVdt = check_divergence_staggered(img[i,:,:,:], res)
                else:
                    divergence = 0.0
                    dUdt       = 0.0
                    dVdt       = 0.0
                div[reslog] = divergence
                momU[reslog] = dUdt
                momV[reslog] = dVdt

                axs[i*4+0].axis('off')
                axs[i*4+1].axis('off')
                axs[i*4+2].axis('off')

                axs[i*4+0].pcolormesh(img[i,0,:,:], cmap='Blues',  edgecolors='k', linewidths=0.1, shading='gouraud')
                axs[i*4+1].pcolormesh(img[i,1,:,:], cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud')
                axs[i*4+2].pcolormesh(img[i,2,:,:], cmap='hot',    edgecolors='k', linewidths=0.1, shading='gouraud')

            else:

                nimg = np.zeros([1, res, res], dtype=DTYPE)
                maxW = np.max(img[i,:,:,:])
                minW = np.min(img[i,:,:,:])
                nimg[0,:,:] = (img[i,:,:,:] - minW)/(maxW - minW)

                nimg = np.uint8(nimg*255)
                nimg = np.transpose(nimg, axes=[1,2,0])

                axs[i].axis('off')
                axs[i].imshow(nimg,cmap='gray')

        fig.savefig('images/image_{:d}x{:d}/it_{:06d}.png'.format(res,res,iteration), bbox_inches='tight', pad_inches=0)
        plt.close('all')


    return div, momU, momV
