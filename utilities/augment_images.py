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
import numpy as np
import sys
import os

from PIL import Image, ImageOps
from matplotlib import cm

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_functions import load_fields

from parameters import OUTPUT_DIM, DTYPE
from functions import nr



N = OUTPUT_DIM
PATH = '../../../data/N256_test_procedures/fields/'
nimg = np.zeros([N, N, 3], dtype=DTYPE)
fimg = np.zeros([N, N, 3], dtype=DTYPE)

files = os.listdir(PATH)
for file in files:
    filename = PATH + file
    tail = filename[-4:]    

    if (tail == '.npz'):
        img = np.zeros([N,N,3], dtype=DTYPE)
        U, V, P, B, C, totTime = load_fields(filename)
        img[:,:,0] = U
        img[:,:,1] = V
        img[:,:,2] = P
        
        # #flip horizontally
        # for ch in range(3):
        #     nimg[:,:,ch] = np.flip(img[:,:,ch], 1)
        # new_tail = "_fliph.npz"
        # filename_new = PATH + file
        # filename_new = filename_new.replace(tail, new_tail)
        # np.savez(filename_new, U=nimg[:,:,0], V=nimg[:,:,1], P=nimg[:,:,2], C=C, B=B, t=totTime)


        # #flip vertically
        # for ch in range(3):
        #     nimg[:,:,ch] = np.flip(img[:,:,ch], 0)
        # new_tail = "_fliphv.npz"
        # filename_new = PATH + file
        # filename_new = filename_new.replace(tail, new_tail)
        # np.savez(filename_new, U=nimg[:,:,0], V=nimg[:,:,1], P=nimg[:,:,2], C=C, B=B, t=totTime)


        #flip horizontally and vertically
        for ch in range(3):
            nimg[:,:,ch] = np.flip(img[:,:,ch],  axis=(1,0))
        new_tail = "_flipv.npz"
        filename_new = PATH + file
        filename_new = filename_new.replace(tail, new_tail)
        np.savez(filename_new, U=nimg[:,:,0], V=nimg[:,:,1], P=nimg[:,:,2], C=C, B=B, t=totTime)

        #random shifts
        ranx = np.random.randint(-N/2+1,N/2-1,3)
        rany = np.random.randint(-N/2+1,N/2-1,3)
        cont=0
        for ii in ranx:
            for jj in rany:

                if (cont%2 == 0):
                    for ch in range(3):
                        fimg[:,:,ch] = np.flip(img[:,:,ch], axis=(1,0))
                else:
                    for ch in range(3):                    
                        fimg[:,:,ch] = img[:,:,ch]

                for ch in range(3):
                    nimg[:,:,ch] = nr(fimg[:,:,ch], ii, jj)

                new_tail = "_shift_" + str(cont) + ".npz"
                filename_new = PATH + file
                filename_new = filename_new.replace(tail, new_tail)
                np.savez(filename_new, U=nimg[:,:,0], V=nimg[:,:,1], P=nimg[:,:,2], C=C, B=B, t=totTime)
                cont=cont+1

    elif (tail == '.png'):
        img = Image.open(filename)

        #flip horizontally
        nimg = ImageOps.Mirror(img)
        new_tail = "_fliph" + tail
        filename_new = PATH + file
        filename_new.replace(tail, new_tail)
        nimg.save(filename_new, nimg)

    print('Done for file ', filename)




# def flip_matrix(A,dir):
#     newA = np.zeros(shape=A.shape, dtype=A.dtype)
#     if (dir=='h'):
#         newA = np.flip(A, 0)
#         # N = len(A[0,:])
#         # for i in range(N):
#         #     newA[:,i] = A[:,N-i-1]
#     elif (dir=='v'):
#         newA = np.flip(A, 1)
#         # N = len(A[:,0])
#         # for j in range(N):
#         #     newA[j,:] = A[N-j-1,:]

#     return newA

# x = np.array([[1,2,3],[4,5,6]])
# print(x)
# a = flip_matrix(x,'h')
# print(a)
# b = flip_matrix(x,'v')
# print(b)

# exit()
