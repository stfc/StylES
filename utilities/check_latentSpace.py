import os
import sys
import scipy as sc

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *
from HIT_2D import L

os.chdir('../')
from MSG_StyleGAN_tf2 import *
from IO_functions import StyleGAN_load_fields
from functions    import gaussian_kernel
os.chdir('./utilities')

from tensorflow.keras.applications.vgg16 import VGG16



# local parameters
USE_WLATENTS    = False   # "LATENTS" consider also mapping, DLATENTS only synthesis
NL              = 1      # number of different latent vectors randomly selected
RANDOMIZE_NOISE = False
TUNE_NOISE      = False
CALCULATE_W     = True
LOAD_FIELD      = True       # load field from DNS solver (via restart.npz file)
FILE_REAL       = "../LES_Solvers/fields/fields_run0_1818te.npz"
WL_IRESTART     = False
WL_CHKP_DIR     = "./wl_checkpoints"
WL_CHKP_PREFIX  = os.path.join(WL_CHKP_DIR, "ckpt")

# clean up and prepare folders
os.system("rm -rf results/plots")
os.system("rm -rf results/fields")
os.system("rm -rf results/uvw")
os.system("rm -rf results/energy")
if (LOAD_FIELD):
    os.system("rm -rf results/plots_org")
    os.system("rm -rf results/fields_org")
    os.system("rm -rf results/uvw_org")
    os.system("rm -rf results/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results/plots")
os.system("mkdir -p results/fields")
os.system("mkdir -p results/uvw")
os.system("mkdir -p results/energy")
os.system("mkdir -p results/plots_org/")
os.system("mkdir -p results/fields_org")
os.system("mkdir -p results/uvw_org")
os.system("mkdir -p results/energy_org")

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(0)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  
N_DNS = OUTPUT_DIM
N_LES = int(2**RES_LOG2_FIL)

P_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
C_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
B_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)

P_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
C_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
B_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)

SIG   = int(N_DNS/N_LES)  # Gaussian (tf and np) filter sigma
DW    = int(N_DNS/N_LES)  # downscaling factor

minMaxUVW = np.zeros((RES_LOG2-3,6), dtype="float32")
minMaxUVW[:,0] = 1.0
minMaxUVW[:,2] = 1.0
minMaxUVW[:,4] = 1.0


with mirrored_strategy.scope():
        
    # Download VGG16 model
    VGG_model         = VGG16(input_shape=(OUTPUT_DIM, OUTPUT_DIM, NUM_CHANNELS), include_top=False, weights='imagenet')
    VGG_features_list = [layer.output for layer in VGG_model.layers]
    VGG_extractor     = tf.keras.Model(inputs=VGG_model.input, outputs=VGG_features_list)


    # loading StyleGAN checkpoint and filter
    checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))


    latents          = tf.keras.Input(shape=[LATENT_SIZE])
    tminMaxUVW       = tf.keras.Input(shape=[6], dtype="float32")

    noise = []
    for ldx in range(G_LAYERS):
        reslog = ldx // 2 + 2
        shape = [1, 2**reslog, 2**reslog]
        noise_in = tf.keras.Input(shape=shape, dtype=DTYPE)
        noise.append(noise_in)        

    lzlatents        = layer_zlatent(latents)
    zlatents         = lzlatents(latents)
    wlatents         = mapping(zlatents)

    lwlatents        = layer_wlatent(wlatents)
    wpdlatents       = lwlatents(wlatents)

    noutputs         = synthesis([wpdlatents, noise], training=False)
    rescale          = layer_rescale(name="layer_rescale")
    outputs, UVW_DNS = rescale(noutputs, tminMaxUVW)
    wl_synthesis     = tf.keras.Model(inputs=[latents, noise, tminMaxUVW], outputs=[outputs, UVW_DNS])



    # # create synthesis model
    # if (USE_WLATENTS):
    #     dlatents         = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
    #     tminMaxUVW       = tf.keras.Input(shape=[6], dtype="float32")
    #     noise = []
    #     for ldx in range(G_LAYERS):
    #         reslog = ldx // 2 + 2
    #         shape = [1, 2**reslog, 2**reslog]
    #         noise_in = tf.keras.Input(shape=shape, dtype=DTYPE)
    #         noise.append(noise_in)        
    #     wlatents         = layer_wlatent(dlatents)
    #     ndlatents        = wlatents(dlatents)
    #     noutputs         = synthesis([ndlatents, noise], training=False)
    #     rescale          = layer_rescale(name="layer_rescale")
    #     outputs, UVW_DNS = rescale(noutputs, tminMaxUVW)
    #     wl_synthesis     = tf.keras.Model(inputs=[dlatents, noise, tminMaxUVW], outputs=[outputs, UVW_DNS])
    # else:
    #     latents          = tf.keras.Input(shape=[LATENT_SIZE])
    #     tminMaxUVW       = tf.keras.Input(shape=[6], dtype="float32")
    #     noise = []
    #     for ldx in range(G_LAYERS):
    #         reslog = ldx // 2 + 2
    #         shape = [1, 2**reslog, 2**reslog]
    #         noise_in = tf.keras.Input(shape=shape, dtype=DTYPE)
    #         noise.append(noise_in)        
    #     wlatents         = layer_zlatent(latents)
    #     nlatents         = wlatents(latents)
    #     dlatents         = mapping(nlatents)
    #     noutputs         = synthesis([dlatents, noise], training=False)
    #     rescale          = layer_rescale(name="layer_rescale")
    #     outputs, UVW_DNS = rescale(noutputs, tminMaxUVW)
    #     wl_synthesis     = tf.keras.Model(inputs=[latents, noise, tminMaxUVW], outputs=[outputs, UVW_DNS])


    # define learnin rate schedule and optimizer
    if (lrDNS_POLICY=="EXPONENTIAL"):
        lr_schedule  = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lrDNS,
            decay_steps=lrDNS_STEP,
            decay_rate=lrDNS_RATE,
            staircase=lrDNS_EXP_ST)
    elif (lrDNS_POLICY=="PIECEWISE"):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrDNS_BOUNDS, lrDNS_VALUES)
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)


    # define checkpoint
    wl_checkpoint = tf.train.Checkpoint(wl_synthesis_0=wl_synthesis)


    # Load latest checkpoint, if restarting
    if (WL_IRESTART):
        status = wl_checkpoint.restore(tf.train.latest_checkpoint(WL_CHKP_DIR))




# add latent space to trainable variables
list_z_trainable_variables = []
for variable in lzlatents.trainable_variables:
    list_z_trainable_variables.append(variable)


list_w_trainable_variables = []
for variable in lwlatents.trainable_variables:
    list_w_trainable_variables.append(variable)

if (not TUNE_NOISE):
    list_DNS_trainable_variables = []
    list_LES_trainable_variables = []

if (USE_WLATENTS):
    for variable in lwlatents.trainable_variables:
        list_DNS_trainable_variables.append(variable)
        if "LES" in variable.name:
            list_LES_trainable_variables.append(variable)
else:
    for variable in lwlatents.trainable_variables:
        list_DNS_trainable_variables.append(variable)
        list_LES_trainable_variables.append(variable)


# print trainable variables
# wl_synthesis.summary()

# print("\n DNS trainable variables are:")
# for variable in list_DNS_trainable_variables:
#     print(variable.name)

# print("\n LES trainable variables are:")
# for variable in list_LES_trainable_variables:
#     print(variable.name)



# define step for finding latent space
@tf.function
def find_latent(latent, noise, minMaxUVW, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions, UVW_DNS = wl_synthesis([latent, noise, minMaxUVW], training=False)
        UVW_LES     = predictions[RES_LOG2_FIL-2]

        # separate DNS fields
        rs = SIG
        U_DNS_t = UVW_DNS[0,0,:,:]
        V_DNS_t = UVW_DNS[0,1,:,:]
        W_DNS_t = UVW_DNS[0,2,:,:]

        U_DNS_t = U_DNS_t[tf.newaxis,:,:,tf.newaxis]
        V_DNS_t = V_DNS_t[tf.newaxis,:,:,tf.newaxis]
        W_DNS_t = W_DNS_t[tf.newaxis,:,:,tf.newaxis]

        # prepare Gaussian Kernel
        gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        gauss_kernel = tf.cast(gauss_kernel, dtype=U_DNS_t.dtype)

        # add padding
        pleft   = 4*rs
        pright  = 4*rs
        ptop    = 4*rs
        pbottom = 4*rs

        U_DNS_t = periodic_padding_flexible(U_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
        V_DNS_t = periodic_padding_flexible(V_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
        W_DNS_t = periodic_padding_flexible(W_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

        # convolve
        fU_t = tf.nn.conv2d(U_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        fV_t = tf.nn.conv2d(V_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        fP_t = tf.nn.conv2d(W_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

        # downscale
        fU = fU_t[0,::DW,::DW,0]
        fV = fV_t[0,::DW,::DW,0]
        fP = fP_t[0,::DW,::DW,0]

        # normalize
        fU = (fU - tf.math.reduce_min(fU))/(tf.math.reduce_max(fU) - tf.math.reduce_min(fU))*two - one
        fV = (fV - tf.math.reduce_min(fV))/(tf.math.reduce_max(fV) - tf.math.reduce_min(fV))*two - one
        fP = (fP - tf.math.reduce_min(fP))/(tf.math.reduce_max(fP) - tf.math.reduce_min(fP))*two - one

        fU = fU[tf.newaxis, tf.newaxis, :, :]
        fV = fV[tf.newaxis, tf.newaxis, :, :]
        fP = fP[tf.newaxis, tf.newaxis, :, :]

        imgB = tf.concat([fU, fV, fP], 1)



        # normalize values
        #loss_fea = VGG_loss(imgA, UVW_DNS, VGG_extractor) 
        #loss_pix_DNS  =  tf.math.reduce_sum(loss_fea[2])
        loss_pix_DNS = tf.math.reduce_mean(tf.math.squared_difference(imgA[0,:,:,:], UVW_DNS[0,:,:,:]))
        loss_pix_LES = 0.0 #tf.math.reduce_mean(tf.math.squared_difference(imgB[0,:,:,:], UVW_LES[0,:,:,:]))
        resDNS       = loss_pix_DNS + loss_pix_LES

        gradients_DNS  = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

    return resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES


@tf.function
def find_latent_step(latent, noise, minMaxUVW, images, list_trainable_variables):
    resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = mirrored_strategy.run(find_latent, args=(latent, noise, minMaxUVW, images, list_trainable_variables))
    return resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES





# define step for finding latent space
@tf.function
def find_latent_w(latent, noise, minMaxUVW, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions, UVW_DNS = wl_synthesis([latent, noise, minMaxUVW], training=False)
        UVW_LES     = predictions[RES_LOG2_FIL-2]

        # separate DNS fields
        rs = SIG
        U_DNS_t = UVW_DNS[0,0,:,:]
        V_DNS_t = UVW_DNS[0,1,:,:]
        W_DNS_t = UVW_DNS[0,2,:,:]

        U_DNS_t = U_DNS_t[tf.newaxis,:,:,tf.newaxis]
        V_DNS_t = V_DNS_t[tf.newaxis,:,:,tf.newaxis]
        W_DNS_t = W_DNS_t[tf.newaxis,:,:,tf.newaxis]

        # prepare Gaussian Kernel
        gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        gauss_kernel = tf.cast(gauss_kernel, dtype=U_DNS_t.dtype)

        # add padding
        pleft   = 4*rs
        pright  = 4*rs
        ptop    = 4*rs
        pbottom = 4*rs

        U_DNS_t = periodic_padding_flexible(U_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
        V_DNS_t = periodic_padding_flexible(V_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
        W_DNS_t = periodic_padding_flexible(W_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

        # convolve
        fU_t = tf.nn.conv2d(U_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        fV_t = tf.nn.conv2d(V_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        fP_t = tf.nn.conv2d(W_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

        # downscale
        fU = fU_t[0,::DW,::DW,0]
        fV = fV_t[0,::DW,::DW,0]
        fP = fP_t[0,::DW,::DW,0]

        # normalize
        fU = (fU - tf.math.reduce_min(fU))/(tf.math.reduce_max(fU) - tf.math.reduce_min(fU))*two - one
        fV = (fV - tf.math.reduce_min(fV))/(tf.math.reduce_max(fV) - tf.math.reduce_min(fV))*two - one
        fP = (fP - tf.math.reduce_min(fP))/(tf.math.reduce_max(fP) - tf.math.reduce_min(fP))*two - one

        fU = fU[tf.newaxis, tf.newaxis, :, :]
        fV = fV[tf.newaxis, tf.newaxis, :, :]
        fP = fP[tf.newaxis, tf.newaxis, :, :]

        imgB = tf.concat([fU, fV, fP], 1)



        # normalize values
        loss_fea = VGG_loss(imgA, UVW_DNS, VGG_extractor) 
        loss_pix_DNS  =  tf.math.reduce_sum(loss_fea[2])
        #loss_pix_DNS = tf.math.reduce_mean(tf.math.squared_difference(imgA[0,:,:,:], UVW_DNS[0,:,:,:]))
        loss_pix_LES = 0.0 #tf.math.reduce_mean(tf.math.squared_difference(imgB[0,:,:,:], UVW_LES[0,:,:,:]))
        resDNS       = loss_pix_DNS + loss_pix_LES

        gradients_DNS  = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

    return resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES


@tf.function
def find_latent_step_w(latent, noise, minMaxUVW, images, list_trainable_variables):
    resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = mirrored_strategy.run(find_latent_w, args=(latent, noise, minMaxUVW, images, list_trainable_variables))
    return resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES










# define step for finding latent space
@tf.function
def find_latent_z(latent, noise, minMaxUVW, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions, UVW_DNS = wl_synthesis([latent, noise, minMaxUVW], training=False)
        UVW_LES     = predictions[RES_LOG2_FIL-2]

        # separate DNS fields
        rs = SIG
        U_DNS_t = UVW_DNS[0,0,:,:]
        V_DNS_t = UVW_DNS[0,1,:,:]
        W_DNS_t = UVW_DNS[0,2,:,:]

        U_DNS_t = U_DNS_t[tf.newaxis,:,:,tf.newaxis]
        V_DNS_t = V_DNS_t[tf.newaxis,:,:,tf.newaxis]
        W_DNS_t = W_DNS_t[tf.newaxis,:,:,tf.newaxis]

        # prepare Gaussian Kernel
        gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        gauss_kernel = tf.cast(gauss_kernel, dtype=U_DNS_t.dtype)

        # add padding
        pleft   = 4*rs
        pright  = 4*rs
        ptop    = 4*rs
        pbottom = 4*rs

        U_DNS_t = periodic_padding_flexible(U_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
        V_DNS_t = periodic_padding_flexible(V_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
        W_DNS_t = periodic_padding_flexible(W_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

        # convolve
        fU_t = tf.nn.conv2d(U_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        fV_t = tf.nn.conv2d(V_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        fP_t = tf.nn.conv2d(W_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

        # downscale
        fU = fU_t[0,::DW,::DW,0]
        fV = fV_t[0,::DW,::DW,0]
        fP = fP_t[0,::DW,::DW,0]

        # normalize
        fU = (fU - tf.math.reduce_min(fU))/(tf.math.reduce_max(fU) - tf.math.reduce_min(fU))*two - one
        fV = (fV - tf.math.reduce_min(fV))/(tf.math.reduce_max(fV) - tf.math.reduce_min(fV))*two - one
        fP = (fP - tf.math.reduce_min(fP))/(tf.math.reduce_max(fP) - tf.math.reduce_min(fP))*two - one

        fU = fU[tf.newaxis, tf.newaxis, :, :]
        fV = fV[tf.newaxis, tf.newaxis, :, :]
        fP = fP[tf.newaxis, tf.newaxis, :, :]

        imgB = tf.concat([fU, fV, fP], 1)



        # normalize values
        loss_fea = VGG_loss(imgA, UVW_DNS, VGG_extractor) 
        loss_pix_DNS  =  tf.math.reduce_sum(loss_fea[2])
        #loss_pix_DNS = tf.math.reduce_mean(tf.math.squared_difference(imgA[0,:,:,:], UVW_DNS[0,:,:,:]))
        loss_pix_LES = 0.0 #tf.math.reduce_mean(tf.math.squared_difference(imgB[0,:,:,:], UVW_LES[0,:,:,:]))
        resDNS       = loss_pix_DNS + loss_pix_LES

        gradients_DNS  = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

    return resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES


@tf.function
def find_latent_step_z(latent, noise, minMaxUVW, images, list_trainable_variables):
    resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = mirrored_strategy.run(find_latent_z, args=(latent, noise, minMaxUVW, images, list_trainable_variables))
    return resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES











# prepare noise for
tf.random.set_seed(0)
noise = []
for ldx in range(G_LAYERS):
    reslog = ldx // 2 + 2
    shape = [BATCH_SIZE, 1, 2**reslog, 2**reslog]
    rnoise = tf.random.uniform(shape=shape, dtype=DTYPE)
    noise.append(rnoise)


# print different fields (to check quality and find 2 different seeds)
for k in range(NL):
    
    # set noise. Note: this noise is different for each latent space instance k. 
    # If RANDOMIZE_NOISE is true, then it will be different in each realization which usually is not the case
    # as it will make the search in the latent space very difficult
    tf.random.set_seed(k)
    noise = []
    for ldx in range(G_LAYERS):
        reslog = ldx // 2 + 2
        shape = [BATCH_SIZE, 1, 2**reslog, 2**reslog]
        rnoise = tf.random.uniform(shape=shape, dtype=DTYPE)
        noise.append(rnoise)


    # load initial flow
    if (LOAD_FIELD):

        if (FILE_REAL.endswith('.npz')):
            
            # load numpy array
            U_DNS, V_DNS, P_DNS, W_DNS, totTime = load_fields(FILE_REAL)
            U_DNS = np.cast[DTYPE](U_DNS)
            V_DNS = np.cast[DTYPE](V_DNS)
            P_DNS = np.cast[DTYPE](P_DNS)
            W_DNS = np.cast[DTYPE](W_DNS)

        elif (FILE_REAL.endswith('.png')):

            # load image
            orig = Image.open(FILE_REAL).convert('RGB')

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

            U_DNS = orig[:,:,0]
            V_DNS = orig[:,:,1]
            W_DNS = orig[:,:,2]

        if (CALCULATE_W):
            W_DNS = find_vorticity(U_DNS, V_DNS)

        #print_fields_1(W_DNS, "Plots_DNS_org.png")
        print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N_DNS, "results/plots_org/Plots_DNS_org.png")
        W_DNS_org = W_DNS


        # write fields and energy spectra of DNS
        DIM_DATA, _ = U_DNS.shape
        closePlot=False
        for kk in range(4, RES_LOG2+1):
            res = 2**kk
            s = int(res/DIM_DATA)
            rs = int(DIM_DATA/res)
            if (s == 1):
                U_DNS_t = U_DNS[:,:]
                V_DNS_t = V_DNS[:,:]
                P_DNS_t = P_DNS[:,:]
                W_DNS_t = W_DNS[:,:]
            else:
                # U_DNS_t = sc.ndimage.gaussian_filter(U_DNS, rs, mode='grid-wrap')
                # V_DNS_t = sc.ndimage.gaussian_filter(V_DNS, rs, mode='grid-wrap')
                # P_DNS_t = sc.ndimage.gaussian_filter(P_DNS, rs, mode='grid-wrap')
                # W_DNS_t = sc.ndimage.gaussian_filter(W_DNS, rs, mode='grid-wrap')

                # U_DNS_t = U_DNS_t[::rs, ::rs]
                # V_DNS_t = V_DNS_t[::rs, ::rs]
                # P_DNS_t = P_DNS_t[::rs, ::rs]
                # W_DNS_t = W_DNS_t[::rs, ::rs]

                # preprare Gaussian Kernel
                gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
                gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

                # convert to tensor
                U_DNS_t = tf.convert_to_tensor(U_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                V_DNS_t = tf.convert_to_tensor(V_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                P_DNS_t = tf.convert_to_tensor(P_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                W_DNS_t = tf.convert_to_tensor(W_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)

                # add padding
                pleft   = 4*rs
                pright  = 4*rs
                ptop    = 4*rs
                pbottom = 4*rs

                U_DNS_t = periodic_padding_flexible(U_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                V_DNS_t = periodic_padding_flexible(V_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                P_DNS_t = periodic_padding_flexible(P_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                W_DNS_t = periodic_padding_flexible(W_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

                # convolve
                U_DNS_t = tf.nn.conv2d(U_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                V_DNS_t = tf.nn.conv2d(V_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                P_DNS_t = tf.nn.conv2d(P_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                W_DNS_t = tf.nn.conv2d(W_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

                # downscale
                U_DNS_t = U_DNS_t[0,::rs,::rs,0].numpy()
                V_DNS_t = V_DNS_t[0,::rs,::rs,0].numpy()
                P_DNS_t = P_DNS_t[0,::rs,::rs,0].numpy()
                W_DNS_t = W_DNS_t[0,::rs,::rs,0].numpy()

            if (CALCULATE_W):
                W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

            minMaxUVW[kk-4,0] = np.max(U_DNS_t)
            minMaxUVW[kk-4,1] = np.min(U_DNS_t)
            minMaxUVW[kk-4,2] = np.max(V_DNS_t)
            minMaxUVW[kk-4,3] = np.min(V_DNS_t)
            minMaxUVW[kk-4,4] = np.max(W_DNS_t)
            minMaxUVW[kk-4,5] = np.min(W_DNS_t)

            filename = "results/plots_org/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
            print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, res, filename)

            filename = "results/fields_org/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
            save_fields(0.0, U_DNS_t, V_DNS_t, P_DNS_t, C_DNS, B_DNS, W_DNS_t, filename)

            filename = "results/energy_org/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
            if (kk==RES_LOG2):
                closePlot=True
            plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

            print("DNS spectrum at resolution " + str(res))

        os.system("mv Energy_spectrum.png results/energy_org/Energy_spectrum_org.png")


        # prepare latent space
        if (USE_WLATENTS):
            zlatent = tf.random.uniform([1, LATENT_SIZE])
            latent  = mapping(zlatent, training=False)
        else:
            latent = tf.random.uniform([1, LATENT_SIZE])
    

        # preprare target image
        kk = RES_LOG2
        res = 2**kk
        tminMaxUVW = tf.convert_to_tensor(minMaxUVW[RES_LOG2-4,:][np.newaxis,:], dtype="float32")
        s = int(res/DIM_DATA)
        rs = int(DIM_DATA/res)
        if (s==1):
            U_DNS_t = U_DNS[:,:]
            V_DNS_t = V_DNS[:,:]
            W_DNS_t = W_DNS[:,:]
        else:            
            U_DNS_t = sc.ndimage.gaussian_filter(U_DNS, rs, mode='grid-wrap')
            V_DNS_t = sc.ndimage.gaussian_filter(V_DNS, rs, mode='grid-wrap')
            W_DNS_t = sc.ndimage.gaussian_filter(W_DNS, rs, mode='grid-wrap')

            U_DNS_t = U_DNS_t[::rs, ::rs]
            V_DNS_t = V_DNS_t[::rs, ::rs]
            W_DNS_t = W_DNS_t[::rs, ::rs]

        if (CALCULATE_W):
            W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)            

        tU_DNS = tf.convert_to_tensor(U_DNS_t)
        tV_DNS = tf.convert_to_tensor(V_DNS_t)
        tW_DNS = tf.convert_to_tensor(W_DNS_t)

        U_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        V_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        W_DNS = tW_DNS[np.newaxis,np.newaxis,:,:]

        imgA = tf.concat([U_DNS, V_DNS, W_DNS], 1)
        #imgA = tf.concat([U_DNS, V_DNS, W_DNS], 1)


        # optimize latent space
        itDNS  = 0
        resDNS = large
        tstart = time.time()
        while (resDNS>tollDNS and itDNS<maxItDNS):

            if (RANDOMIZE_NOISE):
                noise = []
                for ldx in range(G_LAYERS):
                    reslog = ldx // 2 + 2
                    shape = [BATCH_SIZE, 1, 2**reslog, 2**reslog]
                    rnoise = tf.random.uniform(shape=shape, dtype=DTYPE)
                    noise.append(rnoise)

            if (itDNS<maxItDNS/3):
                resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step_z(latent, noise, tminMaxUVW, imgA, list_z_trainable_variables)
            elif (itDNS<maxItDNS/3*2):
                resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step_w(latent, noise, tminMaxUVW, imgA, list_w_trainable_variables)
            else:
                resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step(latent, noise, tminMaxUVW, imgA, list_DNS_trainable_variables)

            # print residuals and fields
            if (itDNS%1000 == 0):

                # find learning rate
                lr = lr_schedule(itDNS)
                with train_summary_writer.as_default():
                    tf.summary.scalar("residuals/loss", resDNS, step=itDNS)
                    tf.summary.scalar("residuals/loss_pix_DNS", loss_pix_DNS, step=itDNS)
                    tf.summary.scalar("residuals/loss_pix_LES", loss_pix_LES, step=itDNS)
                    tf.summary.scalar("lr", lr, step=itDNS)

                # print residuals
                tend = time.time()
                print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  pl_DNS {3:3e}  pl_LES {4:3e}  lr {5:3e} " \
                    .format(tend-tstart, itDNS, resDNS.numpy(), loss_pix_DNS, loss_pix_LES, lr))

                # print fields
                U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
                V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
                W_DNS_t = UVW_DNS[0, 2, :, :].numpy()

                if (CALCULATE_W):
                    W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

                filename = "results/plots/Plots_DNS_fromGAN.png"
                #filename = "results/plots/Plots_DNS_fromGAN" + str(itDNS) + ".png"

                #print_fields_1(W_DNS_t, filename)
                print_fields(U_DNS_t, V_DNS_t, P_DNS, W_DNS_t, N_DNS, filename)

            itDNS = itDNS+1

        # print final values
        lr = lr_schedule(itDNS)
        with train_summary_writer.as_default():
            tf.summary.scalar("residuals/loss", resDNS, step=itDNS)
            tf.summary.scalar("residuals/loss_pix_DNS", loss_pix_DNS, step=itDNS)
            tf.summary.scalar("residuals/loss_pix_LES", loss_pix_LES, step=itDNS)
            tf.summary.scalar("lr", lr, step=itDNS)

        # print residuals
        tend = time.time()
        print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  pl_DNS {3:3e}  pl_LES {4:3e}  lr {5:3e} " \
            .format(tend-tstart, itDNS, resDNS.numpy(), loss_pix_DNS, loss_pix_LES, lr))

        # reprint but only vorticity
        resDNS, predictions, UVW_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step(latent, noise, tminMaxUVW, imgA, list_DNS_trainable_variables)

        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
        W_DNS_t = UVW_DNS[0, 2, :, :].numpy()

        if (CALCULATE_W):
            W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        print_fields_1(W_DNS_t,     "results/plots/Plots_DNS_fromGAN.png", legend=False)
        print_fields_1(W_DNS_org,   "results/plots_org/Plots_DNS_org.png", legend=False)

        # save checkpoint for wl_synthesis
        wl_checkpoint.save(file_prefix = WL_CHKP_PREFIX)

    else:

        # find DNS and LES fields from random input
        tminMaxUVW = tf.convert_to_tensor(minMaxUVW[RES_LOG2-4,:][np.newaxis,:], dtype="float32")
        if (USE_WLATENTS):
            zlatent              = tf.random.uniform([1, LATENT_SIZE])
            dlatents             = mapping(zlatent, training=False)
            predictions, UVW_DNS = wl_synthesis([dlatents, noise, tminMaxUVW], training=False)
        else:
            latents               = tf.random.uniform([1, LATENT_SIZE])
            predictions, UVW_DNS  = wl_synthesis([latents, noise, tminMaxUVW], training=False)



    # print spectrum from filter
    UVW_LES, gauss_image = filter(UVW_DNS, training=False)

    # save internal layer image
    res = 2**RES_LOG2_FIL
    U_t = predictions[RES_LOG2_FIL-2][0, 0, :, :].numpy()
    V_t = predictions[RES_LOG2_FIL-2][0, 1, :, :].numpy()
    W_t = predictions[RES_LOG2_FIL-2][0, 2, :, :].numpy()

    filename = "results/plots/plots_ifil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields(U_t, V_t, P_LES, W_t, res, filename)

    filename = "results/fields/fields_ifil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0.0, U_t, V_t, P_LES, C_LES, B_LES, W_t, filename)

    filename = "results/energy/energy_spectrum_ifil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(U_t, V_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_ifiltered.png")



    # save internal layer image
    res = 2**RES_LOG2_FIL
    U_t = gauss_image[0, 0, :, :].numpy()
    V_t = gauss_image[0, 1, :, :].numpy()
    W_t = gauss_image[0, 2, :, :].numpy()

    filename = "results/plots/plots_gfil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields(U_t, V_t, P_LES, W_t, res, filename)

    filename = "results/fields/fields_gfil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0.0, U_t, V_t, P_LES, C_LES, B_LES, W_t, filename)

    filename = "results/energy/energy_spectrum_gfil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(U_t, V_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_gfiltered.png")



    # save convoluted filtered image
    res = 2**RES_LOG2_FIL
    U_t = UVW_LES[0, 0, :, :].numpy()
    V_t = UVW_LES[0, 1, :, :].numpy()
    W_t = UVW_LES[0, 2, :, :].numpy()

    filename = "results/plots/plots_cfil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields(U_t, V_t, P_LES, W_t, res, filename)

    filename = "results/fields/fields_cfil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0.0, U_t, V_t, P_LES, C_LES, B_LES, W_t, filename)

    filename = "results/energy/energy_spectrum_cfil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(U_t, V_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_cfiltered.png")



    # write fields and energy spectra for each layer
    closePlot=False
    for kk in range(4, RES_LOG2+1):
        UVW_DNS = predictions[kk-2]
        res = 2**kk

        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
        W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
        
        U_DNS_t = two*(U_DNS_t - np.min(U_DNS_t))/(np.max(U_DNS_t) - np.min(U_DNS_t)) - one
        V_DNS_t = two*(V_DNS_t - np.min(V_DNS_t))/(np.max(V_DNS_t) - np.min(V_DNS_t)) - one
        W_DNS_t = two*(W_DNS_t - np.min(W_DNS_t))/(np.max(W_DNS_t) - np.min(W_DNS_t)) - one

        U_DNS_t = (U_DNS_t+one)*(minMaxUVW[kk-4,0]-minMaxUVW[kk-4,1])/two + minMaxUVW[kk-4,1]
        V_DNS_t = (V_DNS_t+one)*(minMaxUVW[kk-4,2]-minMaxUVW[kk-4,3])/two + minMaxUVW[kk-4,3]
        W_DNS_t = (W_DNS_t+one)*(minMaxUVW[kk-4,4]-minMaxUVW[kk-4,5])/two + minMaxUVW[kk-4,5]

        if (CALCULATE_W):
            W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        filename = "results/plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_1(W_DNS_t, filename)

        filename = "results/fields/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
        save_fields(0.0, U_DNS_t, V_DNS_t, P_DNS, C_DNS, B_DNS, W_DNS_t, filename)

        filename = "results/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
        if (kk==RES_LOG2):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

        print("From GAN spectrum at resolution " + str(res))


    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_fromGAN.png")

    print ("done lantent " + str(k))








#--------------------extra pieces

        # # find closest latent space
        # itDNS  = 0
        # resDNS = large
        # tstart = time.time()
        # while (resDNS>tollDNS and itDNS<1000):

        #     # prepare latent space
        #     if (USE_WLATENTS):
        #         zlatent     = tf.random.uniform([1, LATENT_SIZE])
        #         latent      = mapping(zlatent, training=False)
        #         save_latent = mapping(zlatent, training=False)
        #     else:
        #         latent      = tf.random.uniform([1, LATENT_SIZE])
        #         save_latent = tf.random.uniform([1, LATENT_SIZE])

        #     predictions, UVW_DNS = wl_synthesis([latent, minMaxUVW], training=False)
        #     loss_fea = VGG_loss(imgA, UVW_DNS, VGG_extractor) 

        #     # print residuals and fields
        #     if (loss_fea[2]<resDNS):

        #         # save latent space
        #         save_latent = latent

        #         resDNS = loss_fea[2].numpy()

        #         # print residuals
        #         tend = time.time()
        #         print("Latent search  iterations:  time {0:3f}   it {1:3d}  loss_fea {2:3e}" \
        #             .format(tend-tstart, itDNS, resDNS))

        #         # print fields
        #         U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        #         V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
        #         W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
        #         if (CALCULATE_W):  
        #             W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        #         #filename = "results/plots/Plots_DNS_fromGAN.png"
        #         filename = "results/plots/Plots_DNS_fromGAN" + str(itDNS) + ".png"

        #         #print_fields_1(W_DNS_t, filename)
        #         print_fields(U_DNS_t, V_DNS_t, P_DNS, W_DNS_t, N_DNS, filename)

        #     itDNS = itDNS+1


        # latent = save_latent
        # filename = "results/plots/Plots_DNS_fromGAN" + str(itDNS) + ".png"
        # print_fields(U_DNS_t, V_DNS_t, P_DNS, W_DNS_t, N_DNS, filename)
        # exit()