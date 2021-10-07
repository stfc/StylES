import tensorflow as tf
import numpy as np
import scipy as sc

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *

#-------------------------------------------- local functions

# roll (this is also in LES_functions, but cannot be used with CuPy)
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))



#-------------------------------------------- dataset

# variable used during data loading
U = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
P = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)
V = np.zeros([OUTPUT_DIM,OUTPUT_DIM], dtype=DTYPE)


# define data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
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
  
    #resize the image to the desired size.
    img_out = []
    for res in range(2, RES_LOG2 + 1):
        r_img = tf.image.resize(img, [2**res, 2**res])
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

    img_out = []
    for res in range(RES_LOG2-1):
        pow2 = 2**(res+2)
        data = np.zeros([3, pow2, pow2], dtype=DTYPE)
        s = pow2/OUTPUT_DIM
        data[0,:,:] = sc.ndimage.interpolation.zoom(U, s, order=3, mode='wrap')
        data[1,:,:] = sc.ndimage.interpolation.zoom(V, s, order=3, mode='wrap')
        data[2,:,:] = sc.ndimage.interpolation.zoom(P, s, order=3, mode='wrap')
        img_out.append(data)

    return img_out

def process_path_numpy_arrays(path):
    img = tf.numpy_function(StyleGAN_load_fields, \
        [path], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
    return img



# define dataset. Set num_parallel_calls so multiple images are loaded/processed in parallel.
if (READ_NUMPY_ARRAYS):
    labeled_ds = list_ds.map(process_path_numpy_arrays, num_parallel_calls=AUTOTUNE)
else:
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)



# prepare training dataset
def prepare_for_training(ds, cache=True, batch_size=0, shuffle_buffer_size=BUFFER_SIZE, augment=False):

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
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

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


def check_divergence(img, pow2):

    # initialize arrays
    U = img[0,:,:]
    V = img[1,:,:]
    P = img[2,:,:]
    iNN  = 1.0e0/(OUTPUT_DIM*OUTPUT_DIM)
    dl = L/pow2
    A = dl
    Dc = nu/dl*A

    # find Rhie-Chow interpolation (PWIM)
    Ue = hf*(U + cr(U, 1, 0))
    Vn = hf*(V + cr(V, 1, 0))

    Fw = A*rho*cr(Ue, -1, 0)
    Fe = A*rho*Ue
    Fs = A*rho*cr(Vn, 0, -1)
    Fn = A*rho*Vn

    Aw = Dc + hf*(np.abs(Fw) + Fw)
    Ae = Dc + hf*(np.abs(Fe) - Fe)
    As = Dc + hf*(np.abs(Fs) + Fs)
    An = Dc + hf*(np.abs(Fn) - Fn)
    Ao = rho*A*dl/delt

    Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
    iApM = 1.e0/Ap

    deltpX1 = hf*(cr(P, 1, 0) - cr(P, -1, 0))
    deltpX2 = hf*(cr(P, 2, 0) - P)    
    deltpX3 = (P - cr(P,  1, 0))

    deltpY1 = hf*(cr(P, 0, 1) - cr(P, 0, -1))
    deltpY2 = hf*(cr(P, 0, 2) - P)
    deltpY3 = (P - cr(P, 0,  1))

    Ue = hf*(cr(U, 1, 0) + U)                 \
        + hf*deltpX1*iApM*A                   \
        + hf*deltpX2*cr(iApM, 1, 0)*A         \
        + hf*deltpX3*(cr(iApM, 1, 0) + iApM)*A

    Vn = hf*(cr(V, 0, 1) + V)                 \
        + hf*deltpY1*iApM*A                   \
        + hf*deltpY2*cr(iApM, 0, 1)*A         \
        + hf*deltpY3*(cr(iApM, 0, 1) + iApM)*A

    # check divergence
    div = rho*A*np.sum(np.abs(cr(Ue, -1, 0) - Ue + cr(Vn, 0, -1) - Vn))
    div = div*iNN

    # find dU/dt term
    sU = - hf*(cr(P, 1, 0) - cr(P, -1, 0))*A
    dUdt = np.sum(np.abs(sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1)))
    dUdt = dUdt*iNN

    # find dV/dt term
    sV = - hf*(cr(P, 0, 1) - cr(P, 0, -1))*A
    dVdt = np.sum(np.abs(sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1)))
    dVdt = dVdt*iNN

    return div, dUdt, dVdt


def check_divergence_staggered(img, pow2):
    
    # initialize arrays
    U = img[0,:,:]
    V = img[1,:,:]
    P = img[2,:,:]
    iNN  = 1.0e0/(OUTPUT_DIM*OUTPUT_DIM)
    dl = L/pow2
    A = dl
    Dc = nu/dl*A

    # check divergence
    div = rho*A*np.sum(np.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
    div = div*iNN

    # x-direction
    Fw = A*rho*hf*(U            + cr(U, -1, 0))
    Fe = A*rho*hf*(cr(U,  1, 0) + U           )
    Fs = A*rho*hf*(V            + cr(V, -1, 0))
    Fn = A*rho*hf*(cr(V,  0, 1) + cr(V, -1, 1))

    Aw = Dc + hf*(np.abs(Fw) + Fw)
    Ae = Dc + hf*(np.abs(Fe) - Fe)
    As = Dc + hf*(np.abs(Fs) + Fs)
    An = Dc + hf*(np.abs(Fn) - Fn)
    dUdt = np.sum(np.abs(Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1) - (P - cr(P, -1, 0))*A))
    dUdt = dUdt*iNN

    # y-direction
    Fw = A*rho*hf*(U             + cr(U, 0, -1))
    Fe = A*rho*hf*(cr(U,  1,  0) + cr(U, 1, -1))
    Fs = A*rho*hf*(cr(V,  0, -1) + V           )
    Fn = A*rho*hf*(V             + cr(V, 0,  1))

    Aw = Dc + hf*(np.abs(Fw) + Fw)
    Ae = Dc + hf*(np.abs(Fe) - Fe)
    As = Dc + hf*(np.abs(Fs) + Fs)
    An = Dc + hf*(np.abs(Fn) - Fn)
    dVdt = np.sum(np.abs(Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1) - (P - cr(P, 0, -1))*A))
    dVdt = dVdt*iNN

    return div, dUdt, dVdt


def generate_and_save_images(mapping_ave, synthesis_ave, input, iteration):
    dlatents    = mapping_ave(input, training=False)
    predictions = synthesis_ave(dlatents, training=False)

    div  = np.zeros(RES_LOG2-1)
    momU = np.zeros(RES_LOG2-1)
    momV = np.zeros(RES_LOG2-1)
    for res in range(RES_LOG2-1):
        pow2 = 2**(res+2)
        nr = np.int(np.sqrt(NEXAMPLES))
        nc = np.int(NEXAMPLES/nr)
        # to maintain a fixed figure size
        dpi = 367
        fig, axs = plt.subplots(nr,nc, figsize=(1, 1), dpi=dpi, squeeze=False, frameon=False, tight_layout=True)
        #fig, axs = plt.subplots(nr,nc, squeeze=False)
        #plt.subplots_adjust(wspace=0.01, hspace=0.01)
        axs = axs.ravel()
        img = predictions[res]

        # save the highest dimension and first image of the batch as numpy array
        if (pow2==OUTPUT_DIM and SAVE_NUMPY_ARRAYS):
            StyleGAN_save_fields(iteration, img[0,0,:,:], img[0,1,:,:], img[0,2,:,:])

        for i in range(NEXAMPLES):

            #divergence, dUdt, dVdt = check_divergence(img[i,:,:,:], pow2)
            divergence, dUdt, dVdt = check_divergence_staggered(img[i,:,:,:], pow2)
            div[res] = divergence
            momU[res] = dUdt
            momV[res] = dVdt

            # show image
            maxU = np.max(img[i,0,:,:])
            minU = np.min(img[i,0,:,:])
            maxV = np.max(img[i,1,:,:])
            minV = np.min(img[i,1,:,:])
            maxP = np.max(img[i,2,:,:])
            minP = np.min(img[i,2,:,:])
            imax = max(maxU, maxV, maxP)
            imin = min(minU, minV, minP)
            nimg = np.uint8((img[i,:,:,:] - imin)/(imax - imin)*255)
            if (NUM_CHANNELS>1):
                nimg = np.transpose(nimg, axes=[1,2,0])
            axs[i].axis('off')
            if (NUM_CHANNELS==1):
                axs[i].imshow(nimg,cmap='gray')
            else:
                axs[i].imshow(nimg)

        fig.savefig('images/image_{:d}x{:d}/it_{:06d}.png'.format(pow2,pow2,iteration), bbox_inches='tight', pad_inches=0)
        plt.close('all')

    return div, momU, momV
