
Style Eddy Simulation (StylES) is a procedure to simulate plasma physics turbulence. Based on the use of Generative Adversarial Network ([SyleGAN](https://github.com/NVlabs/stylegan)) it reconstructs the non linear terms necessary for the time integration in a Large Eddy Simulation (LES).


</br>

# Description
The idea is to train StyleGAN on DNS data and then use the GAN generator as deconvolution operator for LES. More details are in the following papers:
- J. Castagna and F. Schiavello, ACM PASC23: [**StyleGAN as Deconvolution operator for Large Eddy Simulation**](https://dl.acm.org/doi/abs/10.1145/3592979.3593404) (2023).
- J. Castagna et al., Physics of Plasma: [**StyleGAN as an AI Deconvolution Operator for Large Eddy Simulations of Turbulent Plasma Equations in BOUT++**](https://pubs.aip.org/aip/pop/article/31/3/033902/3278254/StyleGAN-as-an-AI-deconvolution-operator-for-large) (2024).


</br>

# Requirements
We use TensorFlow >= 2.10 installed via pip (see https://www.tensorflow.org/install/pip). We suggest to create a virtual enviroment using the latest list of python module used, available in the **requirements.txt** file.
For the GPU version, make sure you have the following requirements:

cudatoolkit=11.2\
cudnn=8.1.0

which you can easily install via conda.
You will also need to download the TurboGenPY repo from https://github.com/saadgroup/TurboGenPY.git to find the energy spectra. Once cloned (at same directory level of StylES) modify the files using the patch file **patch_TurboGenPY.patch**.

</br>

# Quick start
You can quickly test StylES with BOUT++ using the weights for an already trained StyleGAN according to the following database:

|    Case     |    NxN      |    alpha    |     k       |   $\nu_n=\nu_\zeta$     |   Weights   |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|    HW       |1024x1024    |     1.0     |    1.0      |    10-6     | [mHW_N1024](https://zenodo.org/uploads/13253846)            |

and the following steps:

1. donwload BOUT++, checkout bout_with_StylES branch and compile the hasegawa-wakatani-3d test case as follows:
2. compile BOUT++ as follows:
    - *cmake -S . -B build_release -DBOUT_BUILD_EXAMPLES=ON*
    - *cmake --build /path_to_BOUT/BOUT-dev/build_release -j 16*
    - *cmake --build build_release --target hw3d*
3. download the 1024x1024 weights from the link above and save in the folder **/StylES/checkpoints/**
4. generate the restart point using *python create_restart.py* in the **/StylES/bout_interfaces/** folder
5. run the hasegawa-wakatani-3d test case in the folder **/BOUT-dev/build_release/examples/hasegawa-wakatani-3d/** as *./hw3d*
6. generate the results below using the *python convert_netCDF2png.py* from **/StylES/bout_interfaces/**

Using Paraview and the *.vts* files in the **/StylES/bout_interfaces/results/fields/** you can get the following animation of the vorticity field:

**Animation vorticity field**

<img src="./bout_interfaces/results_StylES/animation_plots.gif" width="1000"/>


<!-- # Comparison  StyleGAN
To make a comparison with the DNS data:
1. go to **/BOUT-dev/examples/hasegawa-wakatani/** and set *int pStepStart = 1000000*
2. repeat step 2.b from previous list
3. rename the *results_StylES* folder as *results_StylES_m1* in **/BOUT-dev/build_release/examples/hasegawa-wakatani/data/**
4. modify BOUT.in in **/BOUT-dev/build_release/examples/hasegawa-wakatani/data/** as follows *nx = 1032* and *nx = 1028*
5. repeat step 4 from previous list
6. run *python plot_comparison.py* from **/StylES/bout_interfaces/** -->


# Training StyleGAN
You first need to generate the DNS data using BOUT++ and then you can train StyleGAN:
1. checkout main version of BOUT++
2. generate DNS data and save in a DATASET folder
3. specify the DATASET path in the **/StylES/parameters** file
4. make sure the BUFFER_SIZE is correctly set. For datasets larger than 50GB you need to change the flag **cache** to False in the **/StylES/IO_functions.py** file (and set the BUFFER_SIZE=1). 
5. set the parameter "randomize_noise = True"
6. train styleGAN via *python main.py*. 


# Copyright
Copyright (C): 2023 UKRI-STFC (Hartree Centre)

Author: Jony Castagna, Francesca Schiavello, Josh Williams

Licence: This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see [GNU-licence](https://www.gnu.org/licenses/).
