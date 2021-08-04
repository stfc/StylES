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
import importlib

sys.path.insert(0, '../')

from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
from matplotlib import gridspec
from parameters import *


NIMG = 10              # number of images to use for statistical values
CONVERT_TO_BW = True

def compare_images(imgA, imgB, title):
    # compute the mean squared error and structural similarity

    # index for the images
    imageA = (np.asarray(imgA, dtype=np.float32))/255.0
    imageB = (np.asarray(imgB, dtype=np.float32))/255.0
    m = np.mean((imageA - imageB)**2)   # Mean Square Error
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=[4, 4, 4, 1])

    # show first image
    ax = fig.add_subplot(spec[0])
    if (CONVERT_TO_BW):
        plt.imshow(imgA, cmap = plt.cm.gray)
    else:
        plt.imshow(imgA)
    plt.axis("off")
    plt.title("Real")

    # show the second image
    ax = fig.add_subplot(spec[1])
    if (CONVERT_TO_BW):
        plt.imshow(imgB, cmap = plt.cm.gray)
    else:
        plt.imshow(imgB)
    plt.axis("off")
    plt.title("StyleLES")

    # show the third image
    ax = fig.add_subplot(spec[2])
    diff = (imageA - imageB)
    maxDiff = np.max(diff)
    minDiff = np.min(diff)
    nDiff = (diff - minDiff)/(maxDiff-minDiff)
    img_diff = Image.fromarray(nDiff)
    img_diff = img_diff.convert("L")
    plt.imshow(img_diff)
    plt.axis("off")
    plt.title("min-max diff [-1,1]")

    # show the colormap
    cax = fig.add_subplot(spec[3])    
    plt.colorbar(cax=cax)
    plt.clim(minDiff,maxDiff)
    plt.suptitle("Statistical differences MSE: %.4f, SSIM: %.4f" % (m, s))

    fig.savefig(title)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)



# load the images -- the orig, the orig + fake
orig = Image.open("./../testloop/data/2D_HIT.png")

# convert to black and white
orig = orig.convert("L") 

# remove white spaces
orig = trim(orig)

# resize images
orig = orig.resize((OUTPUT_DIM,OUTPUT_DIM))

#load fake images
for i in range(NIMG, 0, -1):
    val = TOT_ITERATIONS - IMAGES_EVERY*(i-1)
    filename = "./../images/image_{:d}x{:d}/it_{:06d}.png".format(OUTPUT_DIM, OUTPUT_DIM, val)
    temp = Image.open(filename)
    temp = temp.convert("L")
    temp = trim(temp)
    temp = temp.resize((OUTPUT_DIM,OUTPUT_DIM))
    atemp = np.asarray(temp, dtype=np.float32)    
    if (i==NIMG):
        afake = atemp
    else:
        afake = afake + atemp

if (NIMG>0):
    afake = afake/NIMG
    fake = Image.fromarray(afake)
    fake = fake.convert("L") 

# compare the images
compare_images(orig, fake, "./../testloop/Real_vs_fake.png")
