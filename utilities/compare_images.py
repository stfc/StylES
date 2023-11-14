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
from parameters import DTYPE, OUTPUT_DIM, NUM_CHANNELS, TESTCASE, RES_LOG2
from IO_functions import StyleGAN_load_fields
os.chdir('./utilities')


#-------------------------------- local variables, initialization and functions
N_DNS = 2**RES_LOG2

FILE_PATH  = "/archive/jcastagna/Fields/HW/fields_N1024_1image/"
FILE_REAL  = FILE_PATH + "fields_run1_time1000.npz"
FILE_STYLE = "./results_latentSpace/fields/fields_lat0_res1024.npz"

os.system("rm Plots_DNS_diff.png")


if (TESTCASE=='HIT_2D'):
    labelR = r'$u$'
    labelG = r'$v$'
    labelB = r'$\omega$'

if (TESTCASE=='HW' or TESTCASE=='mHW'):
    labelR = r'$n$'
    labelG = r'$\nabla |\phi|$'
    labelB = r'$\zeta$'



def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def compare_SSIM(imageA, imageB, title):
    # compute SSIM only

    # index for the images
    m = np.mean((imageA - imageB)**2)   # Mean Square Error
    if (NUM_CHANNELS==1):
        s = ssim(imageA, imageB, multichannel=False)
    else:
        s = ssim(imageA, imageB, multichannel=True, channel_axis=2, data_range=2)

    return s


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity

    # index for the images
    m = np.mean((imageA - imageB)**2)   # Mean Square Error
    if (NUM_CHANNELS==1):
        s = ssim(imageA, imageB, multichannel=False)
    else:
        s = ssim(imageA, imageB, multichannel=True, channel_axis=2, data_range=2)


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

    imageD = imageA - imageB
    print(np.min(imageD), np.max(imageD))
    imageD[:,:,0] = imageD[:,:,0]/(np.max(imageD[:,:,0]) - np.min(imageD[:,:,0]))*100
    imageD[:,:,1] = imageD[:,:,1]/(np.max(imageD[:,:,1]) - np.min(imageD[:,:,1]))*100
    imageD[:,:,2] = imageD[:,:,2]/(np.max(imageD[:,:,2]) - np.min(imageD[:,:,2]))*100
    print(np.min(imageD), np.max(imageD))

    # setup figures
    fig, ax = plt.subplots(3, 3, figsize=(15,15))

    # show DNS image
    sub = ax[0,0] 
    im = sub.imshow(imageA[:,:,0], cmap="Blues", vmin=minU, vmax=maxU)
    sub.axis("off")
    sub.set_title("DNS "+ labelR)
    plt.colorbar(im, ax=sub)

    sub = ax[1,0]
    im = sub.imshow(imageA[:,:,1], cmap="RdBu", vmin=minV, vmax=maxV)
    sub.axis("off")
    sub.set_title("DNS "+ labelG)
    plt.colorbar(im, ax=sub)

    sub = ax[2,0]
    im = sub.imshow(imageA[:,:,2], cmap="hot", vmin=minW, vmax=maxW)
    sub.axis("off")
    sub.set_title("DNS "+ labelB)
    plt.colorbar(im, ax=sub)

    # sub = ax[3,0]
    # im = sub.imshow(imageA, cmap="plasma")
    # sub.axis("off")
    # sub.set_title("DNS")
    # plt.colorbar(im, ax=sub)


    # show the StyleGAN image
    sub = ax[0,1] 
    im = sub.imshow(imageB[:,:,0], cmap="Blues", vmin=minU, vmax=maxU)
    sub.axis("off")
    sub.set_title("StyleGAN " + labelR)
    plt.colorbar(im, ax=sub)

    sub = ax[1,1]
    im = sub.imshow(imageB[:,:,1], cmap="RdBu", vmin=minV, vmax=maxV)
    sub.axis("off")
    sub.set_title("StyleGAN " + labelG)
    plt.colorbar(im, ax=sub)

    sub = ax[2,1]
    im = sub.imshow(imageB[:,:,2], cmap="hot", vmin=minW, vmax=maxW)
    sub.axis("off")
    sub.set_title("StyleGAN " + labelB)
    plt.colorbar(im, ax=sub)

    # sub = ax[3,1]
    # im = sub.imshow(imageB, cmap="plasma")
    # sub.axis("off")
    # sub.set_title("StyleGAN")
    # plt.colorbar(im, ax=sub)


    # show the differences
    sub = ax[0,2] 
    im = sub.imshow(imageD[:,:,0], cmap="jet")
    sub.axis("off")
    sub.set_title("diff " + labelR)
    plt.colorbar(im, ax=sub)

    sub = ax[1,2]
    im = sub.imshow(imageD[:,:,1], cmap="jet", vmin=-1, vmax=1)
    sub.axis("off")
    sub.set_title("diff " + labelG)
    plt.colorbar(im, ax=sub)

    sub = ax[2,2]
    im = sub.imshow(imageD[:,:,2], cmap="jet")
    sub.axis("off")
    sub.set_title("diff " + labelB)
    plt.colorbar(im, ax=sub)

    # sub = ax[3,2]
    # im = sub.imshow((imageD-np.min(imageD))/(np.max(imageD)-np.min(imageD)), cmap="jet")
    # sub.axis("off")
    # sub.set_title("diff")
    # plt.colorbar(im, ax=sub)


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
def load_images(file_real, file_Style):
    if (file_real.endswith('.npz')):

        # load numpy array
        orig = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=DTYPE)
        img_in = StyleGAN_load_fields(file_real)
        orig[:,:,0] = img_in[-1][0,:,:]
        orig[:,:,1] = img_in[-1][1,:,:]
        orig[:,:,2] = img_in[-1][2,:,:]
        orig = np.cast[DTYPE](orig)

        # normalize
        orig[:,:,0] = (orig[:,:,0] - np.min(orig[:,:,0]))/(np.max(orig[:,:,0]) - np.min(orig[:,:,0]))
        orig[:,:,1] = (orig[:,:,1] - np.min(orig[:,:,1]))/(np.max(orig[:,:,1]) - np.min(orig[:,:,1]))
        orig[:,:,2] = (orig[:,:,2] - np.min(orig[:,:,2]))/(np.max(orig[:,:,2]) - np.min(orig[:,:,2]))
        
        # calc grad phi
        V_DNS_org = orig[:,:,1]
                
        DELX = 50.176/1024
        DELY = 50.176/1024
        orig[:,:,1] = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                    + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)

    elif (file_real.endswith('.png')):

        # load image
        orig = Image.open(file_real).convert('RGB')

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
    if (file_Style.endswith('.npz')):

        style = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=DTYPE)
        img_in = StyleGAN_load_fields(file_Style)
        style[:,:,0] = img_in[-1][0,:,:]
        style[:,:,1] = img_in[-1][1,:,:]
        style[:,:,2] = img_in[-1][2,:,:]
        style = np.cast[DTYPE](style)
        
        # normalize
        style[:,:,0] = (style[:,:,0] - np.min(style[:,:,0]))/(np.max(style[:,:,0]) - np.min(style[:,:,0]))
        style[:,:,1] = (style[:,:,1] - np.min(style[:,:,1]))/(np.max(style[:,:,1]) - np.min(style[:,:,1]))
        style[:,:,2] = (style[:,:,2] - np.min(style[:,:,2]))/(np.max(style[:,:,2]) - np.min(style[:,:,2]))
        
        # calc grad phi
        V_DNS_org = style[:,:,1]
                
        DELX = 50.176/1024
        DELY = 50.176/1024
        style[:,:,1] = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                    + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)

    else:

        # load image
        style = Image.open(file_Style).convert('RGB')

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

    return orig, style


# # compute SSIM for all fiels in the folder
# files = os.listdir(FILE_PATH)
# nfiles = len(files)
# minSSIM = 1.0
# maxSSIM = 0.0

# for rr in range(100):
#     for tt in range(501,981,10):
#         curr = "fields_run" + str(rr) + "_time" + str(tt).zfill(3) + ".npz"
#         next = "fields_run" + str(rr) + "_time" + str(tt+20).zfill(3) + ".npz"
#         file_curr = FILE_PATH + curr
#         file_next = FILE_PATH + next
#         print(file_curr, file_next)
#         curr, next = load_images(file_curr, file_next)
#         s = compare_SSIM(curr, next, "results_latentSpace/Plots_DNS_diff.png")
#         minSSIM = min(s, minSSIM)
#         maxSSIM = max(s, maxSSIM)

# print("minSSIM, maxSSIM values are: ", minSSIM, maxSSIM)

# compare the images
orig, style = load_images(FILE_REAL, FILE_STYLE)
compare_images(orig, style, "results_latentSpace/Plots_DNS_diff.png")

