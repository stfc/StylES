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

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../')

from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parameters import OUTPUT_DIM, READ_NUMPY, NUM_CHANNELS, TOT_ITERATIONS, IMAGES_EVERY



#-------------------------------- define parameters
NIMG = 1              # number of images to use for statistical values


#-------------------------------- define functions
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))


def load_fields(filename='restart.npz'):
    data = np.load(filename)
    U = data['U']
    V = data['V']
    P = data['P']

    return U, V, P



def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity

    # index for the images
    m = np.mean((imageA - imageB)**2)   # Mean Square Error
    if (NUM_CHANNELS==1):
        s = ssim(imageA, imageB, multichannel=False)
    else:
        s = ssim(imageA, imageB, multichannel=True)


    # check divergence for real image
    U = imageA[:,:,0]
    V = imageA[:,:,1]
    pow2 = imageA.shape[0]
    iNN  = 1.0e0/(pow2*pow2)
    dl = L/pow2
    A = dl
    div = rho*A*np.sum(np.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
    div = div*iNN
    print("Divergence for real image", div)

    # check divergence for fake image
    U = imageB[:,:,0]
    V = imageB[:,:,1]
    pow2 = imageB.shape[0]
    iNN  = 1.0e0/(pow2*pow2)
    dl = L/pow2
    A = dl
    div = rho*A*np.sum(np.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
    div = div*iNN
    print("Divergence for fake image", div)


    # setup figures
    imgA = np.uint8((imageA - np.min(imageA))/(np.max(imageA) - np.min(imageA))*255)
    imgB = np.uint8((imageB - np.min(imageB))/(np.max(imageB) - np.min(imageB))*255)
    fig = plt.figure(title)
    spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=[10, 10, 10, 1])

    # show first image
    ax = fig.add_subplot(spec[0])
    if (NUM_CHANNELS==1):
        plt.imshow(imgA, cmap = plt.cm.gray)
    else:
        plt.imshow(imgA)
    plt.axis("off")
    plt.title("Real")

    # show the second image
    ax = fig.add_subplot(spec[1])
    if (NUM_CHANNELS==1):
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
    if (minDiff==maxDiff):
        img_diff = Image.fromarray(np.uint8(diff*255))  # identical images!
    else:
        nDiff = (diff - minDiff)/(maxDiff-minDiff)
        img_diff = Image.fromarray(np.uint8(nDiff*255))
    if (NUM_CHANNELS==1):
        img_diff = img_diff.convert("L")
    plt.imshow(img_diff)
    plt.axis("off")
    plt.title("diff")

    # show the colormap
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="10%", pad=0.05)

    cax = fig.add_subplot(spec[3])    
    plt.colorbar(cax=cax)
    plt.clim(minDiff,maxDiff)
    plt.suptitle("Statistical differences MSE: %.4f, SSIM: %.4f" % (m, s))

    fig.savefig(title, bbox_inches='tight', pad_inches=0)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)



#-------------------------------- starts comparison
if (READ_NUMPY):
    orig = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=np.float64)
    orig[:,:,0], orig[:,:,1], orig[:,:,2] = load_fields("../testloop/data/from_solver/restart_N32.npz")
    orig = np.cast[np.float64](orig)
else:
    # load image
    orig = Image.open("../testloop/data/from_solver/uvp_0.png").convert('RGB')

    # convert to black and white, if needed
    if (NUM_CHANNELS==1):
        orig = orig.convert("L")

    # remove white spaces
    #orig = trim(orig)

    # resize images
    orig = orig.resize((OUTPUT_DIM,OUTPUT_DIM))

    # convert to numpy array
    orig = np.asarray(orig, dtype=np.float32)
    orig = orig/255.0


# load fake images
if (READ_NUMPY):

    atemp = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=np.float64)
    for i in range(NIMG, 0, -1):
        val = TOT_ITERATIONS - IMAGES_EVERY*(i-1)
        filename = "./../images/image_{:d}x{:d}/restart_it_{:06d}.npz".format(OUTPUT_DIM, OUTPUT_DIM, val)
        atemp[:,:,0], atemp[:,:,1], atemp[:,:,2] = load_fields("../images/image_32x32/restart_it_000000.npz")
        atemp = np.cast[np.float64](atemp)
        if (i==NIMG):
            ttemp = atemp
        else:
            ttemp = ttemp + atemp
    fake = ttemp/NIMG

else:

    for i in range(NIMG, 0, -1):
        val = TOT_ITERATIONS - IMAGES_EVERY*(i-1)
        filename = "./../images/image_{:d}x{:d}/it_{:06d}.png".format(OUTPUT_DIM, OUTPUT_DIM, val)
        temp = Image.open(filename).convert('RGB')
        if (NUM_CHANNELS==1):
            temp = temp.convert("L")
        #temp = trim(temp)
        temp = temp.resize((OUTPUT_DIM,OUTPUT_DIM))
        atemp = np.asarray(temp, dtype=np.float32)    
        atemp = np.asarray(temp, dtype=np.float32)    
        atemp = np.asarray(temp, dtype=np.float32)    
        if (i==NIMG):
            ttemp = atemp
        else:
            ttemp = ttemp + atemp

    ttemp = ttemp/NIMG
    fake = Image.fromarray(np.uint8(ttemp))
    if (NUM_CHANNELS==1):
        fake = fake.convert("L")
    fake = (np.asarray(fake, dtype=np.float32))/255.0

    



# compare the images
compare_images(orig, fake, "diff.png")

