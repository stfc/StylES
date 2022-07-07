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

sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D')
from HIT_2D import delt

from PIL import Image




#-------------------------------------------- dataset

# variable used during data loading
U = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
V = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
P = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
W = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)


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
    data = np.load(file_path)
    U = data['U']
    V = data['V']
    P = data['P']
    U = np.cast[DTYPE](U)
    V = np.cast[DTYPE](V)
    P = np.cast[DTYPE](P)
    DIM_DATA, _ = U.shape

    # normalize the data
    maxU = np.max(U)
    minU = np.min(U)
    if (maxU!=minU):
        U = 2.0*(U - minU)/(maxU - minU) - 1.0
    else:
        U = U
    
    maxV = np.max(V)
    minV = np.min(V)
    if (maxV!=minV):
        V = 2.0*(V - minV)/(maxV - minV) - 1.0
    else:
        V = V

    maxP = np.max(P)
    minP = np.min(P)
    if (maxP!=minP):
        P = 2.0*(P - minP)/(maxP - minP) - 1.0
    else:
        P = P

    # downscale
    img_out = []
    for reslog in range(RES_LOG2-1):
        res = 2**(reslog+2)
        data = np.zeros([3, res, res], dtype=DTYPE)
        s = res/DIM_DATA
        rs = int(DIM_DATA/res)
        if (rs==1):
            data[0,:,:] = U
            data[1,:,:] = V
            data[2,:,:] = P
        else:
            if (TESTCASE=='HW_xwalls'):
                U_DNS_g = sc.ndimage.gaussian_filter(U, rs, mode=['constant','wrap'])
                V_DNS_g = sc.ndimage.gaussian_filter(V, rs, mode=['constant','wrap'])
                P_DNS_g = sc.ndimage.gaussian_filter(P, rs, mode=['constant','wrap'])
            else:
                U_DNS_g = sc.ndimage.gaussian_filter(U, rs, mode='grid-wrap')
                V_DNS_g = sc.ndimage.gaussian_filter(V, rs, mode='grid-wrap')
                P_DNS_g = sc.ndimage.gaussian_filter(P, rs, mode='grid-wrap')
            data[0,:,:] = U_DNS_g[::rs,::rs]
            data[1,:,:] = V_DNS_g[::rs,::rs]  
            data[2,:,:] = P_DNS_g[::rs,::rs]  

        img_out.append(data)

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
def prepare_for_training(ds, cache=True, batch_size=GLOBAL_BATCH_SIZE, shuffle_buffer_size=BUFFER_SIZE, augment=False):

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
    dl = L/res
    A = dl
    Dc = nu/dl*A

    # find Rhie-Chow interpolation (PWIM)
    Ue = hf*(U + nr(U, 1, 0))
    Vn = hf*(V + nr(V, 1, 0))

    Fw = A*rho*nr(Ue, -1, 0)
    Fe = A*rho*Ue
    Fs = A*rho*nr(Vn, 0, -1)
    Fn = A*rho*Vn

    Aw = Dc + hf*(np.abs(Fw) + Fw)
    Ae = Dc + hf*(np.abs(Fe) - Fe)
    As = Dc + hf*(np.abs(Fs) + Fs)
    An = Dc + hf*(np.abs(Fn) - Fn)
    Ao = rho*A*dl/delt

    Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
    iApM = 1.e0/Ap

    deltpX1 = hf*(nr(P, 1, 0) - nr(P, -1, 0))
    deltpX2 = hf*(nr(P, 2, 0) - P)    
    deltpX3 = (P - nr(P,  1, 0))

    deltpY1 = hf*(nr(P, 0, 1) - nr(P, 0, -1))
    deltpY2 = hf*(nr(P, 0, 2) - P)
    deltpY3 = (P - nr(P, 0,  1))

    Ue = hf*(nr(U, 1, 0) + U)                 \
        + hf*deltpX1*iApM*A                   \
        + hf*deltpX2*nr(iApM, 1, 0)*A         \
        + hf*deltpX3*(nr(iApM, 1, 0) + iApM)*A

    Vn = hf*(nr(V, 0, 1) + V)                 \
        + hf*deltpY1*iApM*A                   \
        + hf*deltpY2*nr(iApM, 0, 1)*A         \
        + hf*deltpY3*(nr(iApM, 0, 1) + iApM)*A

    # check divergence
    div = rho*A*np.sum(np.abs(nr(Ue, -1, 0) - Ue + nr(Vn, 0, -1) - Vn))
    div = div*iNN

    # find dU/dt term
    sU = - hf*(nr(P, 1, 0) - nr(P, -1, 0))*A
    dUdt = np.sum(np.abs(sU + Aw*nr(U, -1, 0) + Ae*nr(U, 1, 0) + As*nr(U, 0, -1) + An*nr(U, 0, 1)))
    dUdt = dUdt*iNN

    # find dV/dt term
    sV = - hf*(nr(P, 0, 1) - nr(P, 0, -1))*A
    dVdt = np.sum(np.abs(sV + Aw*nr(V, -1, 0) + Ae*nr(V, 1, 0) + As*nr(V, 0, -1) + An*nr(V, 0, 1)))
    dVdt = dVdt*iNN

    return div, dUdt, dVdt


def check_divergence_staggered(img, res):
    
    # initialize arrays
    U = img[0,:,:]
    V = img[1,:,:]
    P = img[2,:,:]
    iNN  = 1.0e0/(OUTPUT_DIM*OUTPUT_DIM)
    dl = L/res
    A = dl
    Dc = nu/dl*A

    # check divergence
    div = rho*A*np.sum(np.abs(nr(U, 1, 0) - U + nr(V, 0, 1) - V))
    div = div*iNN

    # x-direction
    Fw = A*rho*hf*(U            + nr(U, -1, 0))
    Fe = A*rho*hf*(nr(U,  1, 0) + U           )
    Fs = A*rho*hf*(V            + nr(V, -1, 0))
    Fn = A*rho*hf*(nr(V,  0, 1) + nr(V, -1, 1))

    Aw = Dc + hf*Fw #hf*(np.abs(Fw) + Fw)
    Ae = Dc - hf*Fe #hf*(np.abs(Fe) - Fe)
    As = Dc + hf*Fs #hf*(np.abs(Fs) + Fs)
    An = Dc - hf*Fn #hf*(np.abs(Fn) - Fn)
    dUdt = np.sum(np.abs(Aw*nr(U, -1, 0) + Ae*nr(U, 1, 0) + As*nr(U, 0, -1) + An*nr(U, 0, 1) - (P - nr(P, -1, 0))*A))
    dUdt = dUdt*iNN

    # y-direction
    Fw = A*rho*hf*(U             + nr(U, 0, -1))
    Fe = A*rho*hf*(nr(U,  1,  0) + nr(U, 1, -1))
    Fs = A*rho*hf*(nr(V,  0, -1) + V           )
    Fn = A*rho*hf*(V             + nr(V, 0,  1))

    Aw = Dc + hf*Fw #hf*(np.abs(Fw) + Fw)
    Ae = Dc - hf*Fe #hf*(np.abs(Fe) - Fe)
    As = Dc + hf*Fs #hf*(np.abs(Fs) + Fs)
    An = Dc - hf*Fn #hf*(np.abs(Fn) - Fn)
    dVdt = np.sum(np.abs(Aw*nr(V, -1, 0) + Ae*nr(V, 1, 0) + As*nr(V, 0, -1) + An*nr(V, 0, 1) - (P - nr(P, 0, -1))*A))
    dVdt = dVdt*iNN

    return div, dUdt, dVdt


def generate_and_save_images(mapping, synthesis, input, iteration):
    dlatents    = mapping(input, training=False)
    predictions = synthesis(dlatents, training=False)

    div  = np.zeros(RES_LOG2-1)
    momU = np.zeros(RES_LOG2-1)
    momV = np.zeros(RES_LOG2-1)

    colors = plt.cm.jet(np.linspace(0,1,4))
    lineColor = colors[0]

    for reslog in range(RES_LOG2-1):
        res = 2**(reslog+2)

        # setup figure size
        nr = NEXAMPLES
        nc = 4
        dpi = 100  # scale to a 512 pixel on a 1024x1024 image
        fig, axs = plt.subplots(nr,nc, figsize=(2.5*nc, 10), dpi=dpi)
        plt.subplots_adjust(wspace=0, hspace=0)
        axs = axs.ravel()
        img = predictions[reslog]

        # save the highest dimension and first image of the batch as numpy array
        if (res==OUTPUT_DIM and SAVE_NUMPY_ARRAYS):
            StyleGAN_save_fields(iteration, img[0,0,:,:], img[0,1,:,:], img[0,2,:,:])

        for i in range(NEXAMPLES):

            if (NUM_CHANNELS == 3):

                # print divergence
                divergence, dUdt, dVdt = check_divergence_staggered(img[i,:,:,:], res)
                div[reslog] = divergence
                momU[reslog] = dUdt
                momV[reslog] = dVdt

                # save image after normalization
                nimg = np.zeros([3, res, res], dtype=DTYPE)

                maxUVW = np.max(img[i,:,:,:])
                minUVW = np.min(img[i,:,:,:])
                nimg = (img[i,:,:,:] - minUVW)/(maxUVW - minUVW)

                nimg = np.uint8(nimg*255)
                nimg = np.transpose(nimg, axes=[2,1,0])

                axs[i*4+0].axis('off')
                axs[i*4+1].axis('off')
                axs[i*4+2].axis('off')
                axs[i*4+3].axis('off')

                # x = list(range(res))
                # hdim = res//2
                # yU = nimg[:,hdim,0]
                # yV = nimg[:,hdim,1]
                # yW = nimg[:,hdim,2]

                # axs[i*4+0].plot(x, yU, linewidth=0.1, color=colors[0])
                # axs[i*4+1].plot(x, yV, linewidth=0.1, color=colors[1])
                # axs[i*4+2].plot(x, yW, linewidth=0.1, color=colors[2])
                # axs[i*4+3].plot(x, yW, linewidth=0.1, color=colors[3])

                # axs[i*4+0].imshow(nimg[res/2,:,0],cmap='Blues')
                # axs[i*4+1].imshow(nimg[res/2,:,1],cmap='Reds_r')
                # axs[i*4+2].imshow(nimg[res/2,:,2],cmap='RdBu')
                # axs[i*4+3].imshow(nimg,cmap='jet')

                axs[i*4+0].imshow(nimg[:,:,0],cmap='Blues')
                axs[i*4+1].imshow(nimg[:,:,1],cmap='Reds_r')
                axs[i*4+2].imshow(nimg[:,:,2],cmap='hot')
                axs[i*4+3].imshow(nimg,cmap='jet')

            else:

                nimg = np.zeros([1, res, res], dtype=DTYPE)
                maxW = np.max(img[i,:,:,:])
                minW = np.min(img[i,:,:,:])
                nimg[0,:,:] = (img[i,:,:,:] - minW)/(maxW - minW)

                nimg = np.uint8(nimg*255)
                nimg = np.transpose(nimg, axes=[2,1,0])

                axs[i].axis('off')
                axs[i].imshow(nimg,cmap='gray')

        fig.savefig('images/image_{:d}x{:d}/it_{:06d}.png'.format(res,res,iteration), bbox_inches='tight', pad_inches=0)
        plt.close('all')


    return div, momU, momV
