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

# Series of utilities used across the project

# 1) Plot differences between two images (from https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/)

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import importlib

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D')

from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
from testcases.HIT_2D.HIT_2D import L, rho

os.chdir('../')
from parameters import DTYPE, OUTPUT_DIM, NUM_CHANNELS
from IO_functions import StyleGAN_load_fields
os.chdir('./utilities')


#-------------------------------- local variables, initialization and functions
FILE_REAL  = "./fields/fields_org_lat_0_res_256.npz"
FILE_STYLE = "./fields/fields_lat_0_res_256.npz"


os.system("rm diff.png")


def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity

    # index for the images
    m = np.mean((imageA - imageB)**2)   # Mean Square Error
    if (NUM_CHANNELS==1):
        s = ssim(imageA, imageB, multichannel=False)
    else:
        s = ssim(imageA, imageB, multichannel=True)


    # check divergence for DNS image
    U = imageA[:,:,0]
    V = imageA[:,:,1]
    res = imageA.shape[0]
    iNN  = 1.0e0/(res*res)
    dl = L/res
    A = dl
    div = rho*A*np.sum(np.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
    div = div*iNN
    print("Divergence for DNS image", div)

    # check divergence for style image
    U = imageB[:,:,0]
    V = imageB[:,:,1]
    res = imageB.shape[0]
    iNN  = 1.0e0/(res*res)
    dl = L/res
    A = dl
    div = rho*A*np.sum(np.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
    div = div*iNN
    print("Divergence for StyleGAN image", div)

    # find differences and min-max
    imageD = imageA - imageB

    minUA = np.min(imageA[:,:,0])
    minUB = np.min(imageB[:,:,0])
    minU  = min(minUA, minUB)
    minVA = np.min(imageA[:,:,1])
    minVB = np.min(imageB[:,:,1])
    minV  = min(minVA, minVB)
    minWA = np.min(imageA[:,:,2])
    minWB = np.min(imageB[:,:,2])
    minW  = min(minWA, minWB)

    maxUA = np.max(imageA[:,:,0])
    maxUB = np.max(imageB[:,:,0])
    maxU  = max(maxUA, maxUB)
    maxVA = np.max(imageA[:,:,1])
    maxVB = np.max(imageB[:,:,1])
    maxV  = max(maxVA, maxVB)
    maxWA = np.max(imageA[:,:,2])
    maxWB = np.max(imageB[:,:,2])
    maxW  = max(maxWA, maxWB)

    # setup figures
    fig, ax = plt.subplots(4, 3, figsize=(15,15))

    # show DNS image
    sub = ax[0,0] 
    im = sub.imshow(imageA[:,:,0], cmap="Blues", vmin=minU, vmax=maxU)
    sub.axis("off")
    sub.set_title("DNS U")
    plt.colorbar(im, ax=sub)

    sub = ax[1,0]
    im = sub.imshow(imageA[:,:,1], cmap="RdBu", vmin=minV, vmax=maxV)
    sub.axis("off")
    sub.set_title("DNS V")
    plt.colorbar(im, ax=sub)

    sub = ax[2,0]
    im = sub.imshow(imageA[:,:,2], cmap="hot", vmin=minW, vmax=maxW)
    sub.axis("off")
    sub.set_title("DNS W")
    plt.colorbar(im, ax=sub)

    sub = ax[3,0]
    im = sub.imshow(imageA, cmap="plasma")
    sub.axis("off")
    sub.set_title("DNS")
    plt.colorbar(im, ax=sub)


    # show the StyleGAN image
    sub = ax[0,1] 
    im = sub.imshow(imageB[:,:,0], cmap="Blues", vmin=minU, vmax=maxU)
    sub.axis("off")
    sub.set_title("StyleGAN U")
    plt.colorbar(im, ax=sub)

    sub = ax[1,1]
    im = sub.imshow(imageB[:,:,1], cmap="RdBu", vmin=minV, vmax=maxV)
    sub.axis("off")
    sub.set_title("StyleGAN V")
    plt.colorbar(im, ax=sub)

    sub = ax[2,1]
    im = sub.imshow(imageB[:,:,2], cmap="hot", vmin=minW, vmax=maxW)
    sub.axis("off")
    sub.set_title("StyleGAN W")
    plt.colorbar(im, ax=sub)

    sub = ax[3,1]
    im = sub.imshow(imageB, cmap="plasma")
    sub.axis("off")
    sub.set_title("StyleGAN")
    plt.colorbar(im, ax=sub)


    # show the differences
    sub = ax[0,2] 
    im = sub.imshow(imageD[:,:,0], cmap="jet")
    sub.axis("off")
    sub.set_title("diff U")
    plt.colorbar(im, ax=sub)

    sub = ax[1,2]
    im = sub.imshow(imageD[:,:,1], cmap="jet")
    sub.axis("off")
    sub.set_title("diff V")
    plt.colorbar(im, ax=sub)

    sub = ax[2,2]
    im = sub.imshow(imageD[:,:,2], cmap="jet")
    sub.axis("off")
    sub.set_title("diff W")
    plt.colorbar(im, ax=sub)

    sub = ax[3,2]
    im = sub.imshow((imageD-np.min(imageD))/(np.max(imageD)-np.min(imageD)), cmap="jet")
    sub.axis("off")
    sub.set_title("diff")
    plt.colorbar(im, ax=sub)


    plt.suptitle("Statistical differences MSE: %.4e, SSIM: %.4e" % (m, s))
    fig.savefig(title, bbox_inches='tight', pad_inches=0)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)



#-------------------------------- starts comparison
if (FILE_REAL.endswith('.npz')):

    # load numpy array
    orig = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=DTYPE)
    img_in = StyleGAN_load_fields(FILE_REAL)
    orig[:,:,0] = img_in[-1][0,:,:]
    orig[:,:,1] = img_in[-1][1,:,:]
    orig[:,:,2] = img_in[-1][2,:,:]
    orig = np.cast[DTYPE](orig)

elif (FILE_REAL.endswith('.png')):

    # load image
    orig = Image.open(FILE_REAL).convert('RGB')

    # convert to black and white, if needed
    if (NUM_CHANNELS==1):
        orig = orig.convert("L")

    # remove white spaces
    #orig = trim(orig)

    # resize images
    orig = orig.resize((OUTPUT_DIM,OUTPUT_DIM))

    # convert to numpy array
    orig = np.asarray(orig, dtype=DTYPE)
    orig = orig/255.0


# load style images
if (FILE_STYLE.endswith('.npz')):

    style = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=DTYPE)
    img_in = StyleGAN_load_fields(FILE_STYLE)
    style[:,:,0] = img_in[-1][0,:,:]
    style[:,:,1] = img_in[-1][1,:,:]
    style[:,:,2] = img_in[-1][2,:,:]
    style = np.cast[DTYPE](style)

else:

    # load image
    style = Image.open(FILE_STYLE).convert('RGB')

    # convert to black and white, if needed
    if (NUM_CHANNELS==1):
        style = style.convert("L")

    # remove white spaces
    #style = trim(style)

    # resize images
    style = style.resize((OUTPUT_DIM,OUTPUT_DIM))

    # convert to numpy array
    style = np.asarray(style, dtype=DTYPE)
    style = style/255.0



# compare the images
compare_images(orig, style, "Plots_DNS_diff.png")

