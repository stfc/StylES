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
import os
import numpy as np

PATH = './fields_single/'
DEST = './fields_UVW/'

def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def find_vorticity(U, V):
    W = ((cr(V, 1, 0)-cr(V, -1, 0)) - (cr(U, 0, 1)-cr(U, 0, -1)))
    return W

def load_fields(filename):
    data = np.load(filename)
    U = data['U']
    V = data['V']
    W = find_vorticity(U,V)
    return U, V, W

def save_fields(U, V, W, filename):
    np.savez(filename, U=U, V=V, W=W)




files = os.listdir(PATH)
nfiles = len(files)
for i,file in enumerate(sorted(files)):
    filename = PATH + file
    filename2 = DEST + file

    U, V, W = load_fields(filename)
    save_fields(U,V,W,filename2)
    
