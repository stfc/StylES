


Style Eddy Simulation (StylES) is a procedure to simulate plasma physics turbulence based on the use of Generative Adversarial Network ([SyleGAN](https://github.com/NVlabs/stylegan)) to reconstruct non linear terms in a Large Eddy Simulation (LES).

Paper is available at [StylES](https://dl.acm.org/doi/10.1145/3592979.3593404)

</br>

# Description
The idea is to train StyleGAN on DNS data and then use the GAN generator as deconvolution operator for LES. More details are in the following papers:
- J. Castagna and F. Schiavello, ACM PASC23: [**StyleGAN as Deconvolution operator for Large Eddy Simulation**](https://dl.acm.org/doi/abs/10.1145/3592979.3593404) (2023).
- J. Castagna et al., Physics of Plasma: [**StyleGAN as an AI Deconvolution Operator for Large Eddy Simulations of Turbulent Plasma Equations in BOUT++**](https://pubs.aip.org/aip/pop/article/31/3/033902/3278254/StyleGAN-as-an-AI-deconvolution-operator-for-large) (2024).

Paper is available at [StylES](https://dl.acm.org/doi/10.1145/3592979.3593404)

</br>

# Requirements
We use TensorFlow >= 2.10 installed via pip (see https://www.tensorflow.org/install/pip). We suggest to create a virtual enviroment using the latest list of python module used, available in the **requirements.txt** file.

For the GPU version, make sure you have the following requirements:

cudatoolkit=11.2\
cudnn=8.1.0

which you can easily install via conda.

You will also need to download the TurboGenPY repo from https://github.com/saadgroup/TurboGenPY.git to find the energy spectra and use  Saad's initialization procedure for HIT_2D. Once cloned (at same directory level of StylES) modify the files using the patch file **patch_TurboGenPY.patch**.

</br>

# Quick start
You can quickly test StylES with BOUT++ using the weights for an already trained StyleGAN according to the following database:

<!-- |    Case     |    NxN      |    alpha    |     k       |   nu=nu     |   Link      |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|    HW       | 256x256     |     1.0     |    0.5      |    10-3     | [HW_N256](https://zenodo.org/records/13253301/files/ckpt-50.[comment]:data-00000-of-00001?download=1)
|    HW       | 512x512     |     1.0     |    1.0      |    10-4     |
|    HW       |1024x1024    |     1.0     |    1.0      |    10-6     | -->

|    Case     |    NxN      |    alpha    |     k       |   nu=nu     |   Link      |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|    HW       |1024x1024    |     1.0     |    1.0      |    10-6     | [mHW_N1024](https://zenodo.org/uploads/13253846)            |

and the following steps:

- 1) donwload BOUT++, checkout bout_with_StylES branch and compile the hasegawa-wakatani as follows:
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.a) cmake -S . -B build_release -DBOUT_BUILD_EXAMPLES=ON
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.b) cmake --build /path_to_BOUT/BOUT-dev/build_release -j 16
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.c) cmake --build build_release --target hasegawa-wakatani
- 2) download the 1024x1024 weights from the link above and save in a folder **/StylES/checkpoints/**
- 3) generate the restart point using *python create_restart.py* in the **/StylES/utility/** folder
- 4) run the hasegawa-wakatani-3d test case in the folder **/BOUT-dev/build_release/examples/hasegawa-wakatani-3d/** as ./hw3d
- 5) generate the results below using the *python convert_netCDF2png.py* from **/StylES/bout_interfaces/**

From paraview, the vts files in the **/StylES/bout_interfaces/results/fields/** will give the following animation:

**Animation vorticity field**

<img src="./bout_interfaces/results_StylES/animation_plots.gif" width="1000"/>

To make a comparison with the DNS data:
- 6) go to **/BOUT-dev/examples/hasegawa-wakatani/** and set *int pStepStart = 1000000*
- 7) repeat step 3.c
- 8) rename the *results_StylES* folder as *results_StylES_m1* in **/BOUT-dev/build_release/examples/hasegawa-wakatani/data/**
- 9) modify BOUT.in in **/BOUT-dev/build_release/examples/hasegawa-wakatani/data/** as follows *nx = 1032* and *nx = 1028*
- 10) repeat step 4
- 11) run *python plot_comparison.py* from **/StylES/bout_interfaces/**

<!-- **Centre line values**

<br />    |  <br />  |  <br />
----------|----------|----------
![image info](./bout_interfaces/results_comparison/DNS_vs_StylES_n.png) | ![image info](./bout_interfaces/results_comparison/DNS_vs_StylES_phi.png) | ![image info](./bout_interfaces/results_comparison/DNS_vs_StylES_vort.png) -->

<!-- **Energy vs time**

<br />    |  <br />
----------|----------
![image info](./bout_interfaces/results_comparison/energy_vs_time.png) | ![image info](./bout_interfaces/results_comparison/enstrophy_vs_time.png)

**Radial ad poloidal fluxes**
<br /> |  <br />
-------|-----------
![image info](./bout_interfaces/results_comparison/radialFlux_vs_time.png) | ![image info](./bout_interfaces/results_comparison/poloidalFlux_vs_time.png)

Animation with comparison to DNS of vorticity $\zeta$

![image info](./bout_interfaces/results_comparison/plot_diffs/animation_diff_vort.gif)
 -->


# Training StyleGAN
You first need to generate the DNS data using BOUT++ and then you can train StyleGAN:
- 1) checkout main version of BOUT++
- 2) generate DNS data and save in a DATASET folder
- 3) specify the DATASET path in the **/StylES/parameters** file
- 4) make sure the BUFFER_SIZE is correctly set. For datasets larger than 50GB you need to change the flag **cache** to False in the **/StylES/IO_functions.py** file (and set the BUFFER_SIZE=1). 
- 5) set the parameter "randomize_noise = True"
- 6) train styleGAN via *python main.py*. 


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
