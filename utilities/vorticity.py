import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib import cm
from PIL import Image

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from parameters import OUTPUT_DIM, TOT_ITERATIONS, IMAGES_EVERY



#-------------------------------------------- local functions

# roll (this is also in LES_functions, but cannot be used with CuPy)
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))



#-------------------------------------------- find vorticity
# load images
NIMG = 10
for i in range(NIMG, 0, -1):
    val = TOT_ITERATIONS - IMAGES_EVERY*(i-1)
    filename = "./../images/image_{:d}x{:d}/it_{:06d}.png".format(OUTPUT_DIM, OUTPUT_DIM, val)
    temp = Image.open(filename).convert('RGB')

    UVW = np.asarray(temp, dtype=np.float32)/255.0
    U = UVW[:, :, 0]
    V = UVW[:, :, 1]

    # find vorticity
    vor = find_vorticity(U, V)
    vor = (vor - np.min(vor))/(np.max(vor) - np.min(vor))
    vor = Image.fromarray(np.uint8(vor*255))
    vor.save("vorticity_" + str(i) + ".png")



