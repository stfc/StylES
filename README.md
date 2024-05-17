Style Eddy Simulation (StylES) is a reconstruction procedure of Direct Numerical Simulation (DNS) results from Large Eddy Simulation (LES).
It is based on traditional LES solvers and Generative Adverserial Networks (GANs), mainly [SyleGAN](https://github.com/NVlabs/stylegan) and [MSG-Style GANs](https://github.com/akanimax/msg-stylegan-tf).

</br>

# Description
The idea is to train StyleGAN on DNS data and then use the GAN generator as deconvolution operator for LES. More details are in the following papers:
- J. Castagna and F. Schiavello, ACM PASC23: **StyleGAN as Deconvolution operator for Large Eddy Simulation** (2023).
- J. Castagna et al., Physics of Plasma: **StyleGAN as an AI Deconvolution Operator for Large Eddy Simulations of Turbulent Plasma Equations in BOUT++** (2024).

</br>

# Requirements
We use TensorFlow >= 2.9 installed via pip (see https://www.tensorflow.org/install/pip). We suggest to create a virtual enviroment using the latest list of python module used, available in the **requirements.txt** file.

For the GPU version, make sure you have the following requirements:

cudatoolkit=11.2\
cudnn=8.1.0

which you can easily install via conda.

You will also need to download the TurboGenPY repo from https://github.com/saadgroup/TurboGenPY.git to find the energy spectra and use  Saad's initialization procedure for HIT_2D. Once cloned (at same directory level of StylES) modify the files using the patch file **patch_TurboGenPY.patch**.

</br>

# Quick start
You can quickly test StylES with BOUT++ using the weights for an already trained StyleGAN according to the following database:

|    Case     |    NxN      |    alpha    |     k       |   nu=nu     |
| ----------- | ----------- | ----------- | ----------- | ----------- |
|    HW       | 256x256     |     1.0     |    0.5      |    10-3     |
|    HW       | 512x512     |     1.0     |    1.0      |    10-4     |
|    HW       |1024x1024    |     1.0     |    1.0      |    10-6     |


and the following steps:

- 1) donwload BOUT++, checkout bout_with_StylES branch and compile the hasegawa-wakatani as follows:
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.a) cmake -S . -B build_release -DBOUT_BUILD_EXAMPLES=ON
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.b) cmake --build /path_to_BOUT/BOUT-dev/build_release -j 16
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.c) cmake --build build_release --target hasegawa-wakatani
- 2) download the 1024x1024 weights from and save in
- 3) generate the restart point usingas *python create_restart.py* in the **/StylES/utility** folder
- 4) run the hasegawa-wakatani test case in the folder **/BOUT-dev/build_release/examples/hasegawa-wakatani/** as ./hasegawa-wakatani
- 5) generate the results below using the *python convert_netCDF2png.py* from **/StylES/bout_interfaces/**

You will obtain the following animatin and plot of energy vs time

To make a comparison with the DNS data:
- 6) go to **/BOUT-dev/examples/hasegawa-wakatani/** and set *int pStepStart = 1000000*
- 7) repeat step 3.c
- 8) save the *results_StylES* folder as *results_StylES_m1* in **/BOUT-dev/build_release/examples/hasegawa-wakatani/data/**
- 9) modify BOUT.in in **/BOUT-dev/build_release/examples/hasegawa-wakatani/data/** as follows *nx = 1032* and *nx = 1028*
- 10) repeat step 4
- 11) run *python plot_comparison.py* from **/StylES/bout_interfaces/**

You will now obtain a series of comparison plots between DNS and StylES.



# Testloop
The following results are obtained via these steps

1) Generate training DNS data using the staggered solver:
 - *python LES_solver_staggered.py* (from **LES_Solvers** folder. This will take ~3days!)

2) Train the StyleGAN
 - *python main.py* (from **root** folder). This will take ~24h and the training should looks like those in the log file reference (open with TensorBoard).

3) Generate new DNS fields to use as target different from the training dataset

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.a) create restart point using
  *python create_restart.py* (from **utility** folder). Change the SEED_RESTART to get a different field which can be seen in **LES_Solvers/plots_restart.png** (ideally you would like something which looks like the fields at $t=545 \tau_e$)\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.b) run *python LES_solver_staggered.py HIT_2D_reconstruction* from **LES_Solvers** folder. This will take ~10min. **Carefull: previous fields files used for the training of the GAN will be overwritten!!**

4) Set the flag RANDOMIZED_NOISE=False (**parameters.py** file)

5) For a given DNS field (default is set to the last field produced during reconstruction, **fields_run0_it10001.npz**), generate the matching field from the StyleGAN via search in the latent space
 - *python check_latentSpace* (from **utility** folder. This will take ~1h due to the low tollerance set)

6) Compare results
 - *python compare_images.py* (from **utility** folder) 
Differences between real DNS field and generated by StyleGAN are in the image below (Plots_DNS_diff.png)
<br/>
<br/>
<br/>

![image info](./utilities/results_latentSpace/Plots_DNS_diff.png)

7) Reconstruction results
- *python check_reconstruction.py* (from **utility** folder) 
Results of the reconstruction are in the image below (uvw_vs_time). This will take ~30min. Results are below
<br/>
<br/>
<br/>

Reconstruction vs time at different tolerances:
![image info](./utilities/results_reconstruction/uvw_vs_time.png)

Profile reconstructed at initial and final time:
![image info](./utilities/results_reconstruction/uvw_vs_x.png)

Energy spectra at initial and final time:
![image info](./utilities/results_reconstruction/Energy_spectrum.png)

Animation with comparison to DNS:
![image info](./utilities/results_reconstruction/animation.gif)

# Copyright
Copyright (C): 2023 UKRI-STFC (Hartree Centre)

Author: Jony Castagna, Francesca Schiavello

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
