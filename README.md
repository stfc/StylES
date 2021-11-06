StylelES is a Style Eddy Simulation based solver for Computational Fluid Dynamic (CFD) simulations.
It is based on traditional LES solvers, like OpenFOAM, and Generative Adverserial Networks (GANs), mainly [SyleGAN](https://github.com/NVlabs/stylegan) and [MSG-Style GANs](https://github.com/akanimax/msg-stylegan-tf).


# Description
The purpose is to capture the Kolmogorov energy cascade training a StyleGAN via Direct Numerical Simulation (DNS). We then extract the Subgrid-Scale model for the LES.


# Requirements
We use TensorFlow 2.2 via conda. We suggest to innstall conda 4.8.3 (or above) and the
requirements creating a conda enviroment as follows (change the myenv name as you wish):

conda create -n myenv --file package-list.txt

For the GPU version, make sure you have the following requirements:

- NVIDIA GPU drivers —> CUDA 10.1 requires 418.x or higher.


# Testloop
We don't have a proper testloop yet, but the following results are obtained via these steps

1) Generate training DNS data using the staggered solver:
 - *python LES_solver_staggered.py* (from **LES_Solvers** folder. This will take ~5h)

2) Train the StyleGAN
 - *python LES_solver_staggered.py* (from **root** folder). This will take another ~2h and the training should looks like those in the log file reference (open with TensorBoard). The following divergence values are obtained:

Total divergencies, dUdt and dVdt for each resolution:
   4x   4:   6.329307e-06   1.010620e-05   7.586500e-06\
   8x   8:   1.008767e-05   9.734142e-06   1.182661e-05\
  16x  16:   3.007163e-05   2.171808e-05   2.683089e-05\
  32x  32:   3.953503e-05   4.594382e-05   6.043978e-05\
  64x  64:   5.004367e-05   7.213454e-05   8.559941e-05\
 128x 128:   5.148542e-05   1.129339e-04   1.311764e-04\
 256x 256:   5.143995e-05   3.776180e-04   3.863021e-04

3) For a given DNS field, generate the matching field from the StyleGAN as search in the latent space
 - *python check_latentSpace* (from **utility** folder. This will take ~1h)

4) Compare results
 - *python compare_images.py* (from **utility** folder) 

![image info](./testloop/result_divergence.png)
