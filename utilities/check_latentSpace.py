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
USE_DLATENTS   = True   # "LATENTS" consider also mapping, DLATENTS only synthesis
NL             = 10        # number of different latent vectors randomly selected
TUNE_NOISE     = False
LOAD_FIELD     = False       # load field from DNS solver (via restart.npz file)
USE_VGG        = False
FILE_REAL      = "../../../data/HW_timeIntegration/fields/fields_200.npz"
WL_IRESTART    = False
WL_CHKP_DIR    = "./wl_checkpoints"
WL_CHKP_PREFIX = os.path.join(WL_CHKP_DIR, "ckpt")

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

N_DNS = 2**RES_LOG2
N_LES = 2**RES_LOG2_FIL

den_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
phi_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
vor_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
zero_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)

den_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
phi_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
vor_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
zero_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)

SIG   = int(N_DNS/N_LES)  # Gaussian (tf and np) filter sigma
DW    = int(N_DNS/N_LES)  # downscaling factor
minMaxUVP = np.zeros((RES_LOG2-3,6), dtype="float32")
minMaxUVP[:,0] = 1.0
minMaxUVP[:,2] = 1.0
minMaxUVP[:,4] = 1.0



with mirrored_strategy.scope():
        

    # Download VGG16 model
    VGG_model         = VGG16(input_shape=(OUTPUT_DIM, OUTPUT_DIM, NUM_CHANNELS), include_top=False, weights='imagenet')
    VGG_features_list = [layer.output for layer in VGG_model.layers]
    VGG_extractor     = tf.keras.Model(inputs=VGG_model.input, outputs=VGG_features_list)


    # Download VGG16 model
    VGG_model_LES         = VGG16(input_shape=(N_LES, N_LES, NUM_CHANNELS), include_top=False, weights='imagenet')
    VGG_features_list_LES = [layer.output for layer in VGG_model_LES.layers]
    VGG_extractor_LES     = tf.keras.Model(inputs=VGG_model_LES.input, outputs=VGG_features_list_LES)


    # loading StyleGAN checkpoint and filter
    checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))


    # create variable synthesis model
    if (USE_DLATENTS):
        dlatents         = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
        tminMaxUVP       = tf.keras.Input(shape=[6], dtype="float32")
        wlatents         = layer_wlatent(dlatents)
        ndlatents        = wlatents(dlatents)
        noutputs         = synthesis(ndlatents, training=False)
        rescale          = layer_rescale(name="layer_rescale")
        outputs, UVP_DNS = rescale(noutputs, tminMaxUVP)
        wl_synthesis     = tf.keras.Model(inputs=[dlatents, tminMaxUVP], outputs=[outputs, UVP_DNS])
    else:
        latents          = tf.keras.Input(shape=[LATENT_SIZE])
        tminMaxUVP       = tf.keras.Input(shape=[6], dtype="float32")
        wlatents         = layer_zlatent(latents)
        nlatents         = wlatents(latents)
        dlatents         = mapping(nlatents)
        noutputs         = synthesis(dlatents, training=False)
        rescale          = layer_rescale(name="layer_rescale")
        outputs, UVP_DNS = rescale(noutputs, tminMaxUVP)
        wl_synthesis     = tf.keras.Model(inputs=[latents, tminMaxUVP], outputs=[outputs, UVP_DNS])

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
    wl_checkpoint = tf.train.Checkpoint(wl_synthesis=wl_synthesis)


    # Load latest checkpoint, if restarting
    if (WL_IRESTART):
        status = wl_checkpoint.restore(tf.train.latest_checkpoint(WL_CHKP_DIR))




# add latent space to trainable variables
if (not TUNE_NOISE):
    list_DNS_trainable_variables = []
    list_LES_trainable_variables = []

for variable in wlatents.trainable_variables:
    list_DNS_trainable_variables.append(variable)
    #if "LES" in variable.name:
    list_LES_trainable_variables.append(variable)

print("\nDNS variables:")
for variable in list_DNS_trainable_variables:
    print(variable.name)

print("\nLES variables:")
for variable in list_LES_trainable_variables:
    print(variable.name)



# define step for finding latent space
@tf.function
def find_latent(latent, minMaxUVP, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions, UVP_DNS = wl_synthesis([latent, minMaxUVP], training=False)

        # find losses
        if (USE_VGG):
            loss_fea     = VGG_loss(imgA, UVP_DNS, VGG_extractor) 
            loss_pix_DNS = 0.0
            loss_pix_LES = 0.0
            resDNS       = tf.math.reduce_sum(loss_fea)
        else:
            loss_pix_DNS = tf.math.reduce_mean(tf.math.squared_difference(imgA[0,:,:,:], UVP_DNS[0,:,:,:]))
            loss_pix_DNS = loss_pix_DNS/tf.math.reduce_mean(imgA[0,:,:,:]**2)
            loss_pix_LES = 0.0
            resDNS       = loss_pix_DNS + loss_pix_LES

        gradients_DNS  = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

    return resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES


@tf.function
def find_latent_step(latent, minMaxUVP, images, list_trainable_variables):
    resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES = mirrored_strategy.run(find_latent, args=(latent, minMaxUVP, images, list_trainable_variables))
    return resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES


# print different fields (to check quality and find 2 different seeds)
for k in range(NL):
    
    # load initial flow
    tf.random.set_seed(k)
    if (LOAD_FIELD):

        if (FILE_REAL.endswith('.npz')):
            
            # load numpy array
            den_DNS, phi_DNS, vor_DNS, _, _, totTime = load_fields(FILE_REAL)
            den_DNS = np.cast[DTYPE](den_DNS)
            phi_DNS = np.cast[DTYPE](phi_DNS)
            vor_DNS = np.cast[DTYPE](vor_DNS)

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

            den_DNS = orig[:,:,0]
            phi_DNS = orig[:,:,1]
            vor_DNS = orig[:,:,2]

        #print_fields(den_DNS, phi_DNS, vor_DNS, W_DNS, N_DNS, "results/plots_org/Plots_DNS_org.png")
        print_fields_1(vor_DNS, "Plots_DNS_org.png")
        vor_DNS_org = vor_DNS


        # write fields and energy spectra of DNS
        DIM_DATA, _ = den_DNS.shape
        closePlot=False
        for kk in range(4, RES_LOG2+1):
            res = 2**kk
            s = int(res/DIM_DATA)
            rs = int(DIM_DATA/res)
            if (s == 1):
                den_DNS_t = den_DNS[:,:]
                phi_DNS_t = phi_DNS[:,:]
                vor_DNS_t = vor_DNS[:,:]
            else:
                # den_DNS_t = sc.ndimage.gaussiaden_filter(den_DNS, rs, mode='grid-wrap')
                # phi_DNS_t = sc.ndimage.gaussiaden_filter(phi_DNS, rs, mode='grid-wrap')
                # vor_DNS_t = sc.ndimage.gaussiaden_filter(vor_DNS, rs, mode='grid-wrap')

                # den_DNS_t = den_DNS_t[::rs, ::rs]
                # phi_DNS_t = phi_DNS_t[::rs, ::rs]
                # vor_DNS_t = vor_DNS_t[::rs, ::rs]

                # preprare Gaussian Kernel
                gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
                gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

                # convert to tensor
                den_DNS_t = tf.convert_to_tensor(den_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                phi_DNS_t = tf.convert_to_tensor(phi_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                vor_DNS_t = tf.convert_to_tensor(vor_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)

                # add padding
                pleft   = 4*rs
                pright  = 4*rs
                ptop    = 4*rs
                pbottom = 4*rs

                den_DNS_t = periodic_padding_flexible(den_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                phi_DNS_t = periodic_padding_flexible(phi_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                vor_DNS_t = periodic_padding_flexible(vor_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

                # convolve
                den_DNS_t = tf.nn.conv2d(den_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                phi_DNS_t = tf.nn.conv2d(phi_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                vor_DNS_t = tf.nn.conv2d(vor_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

                # downscale
                den_DNS_t = den_DNS_t[0,::rs,::rs,0].numpy()
                phi_DNS_t = phi_DNS_t[0,::rs,::rs,0].numpy()
                vor_DNS_t = vor_DNS_t[0,::rs,::rs,0].numpy()

            minMaxUVP[kk-4,0] = np.max(den_DNS_t)
            minMaxUVP[kk-4,1] = np.min(den_DNS_t)
            minMaxUVP[kk-4,2] = np.max(phi_DNS_t)
            minMaxUVP[kk-4,3] = np.min(phi_DNS_t)
            minMaxUVP[kk-4,4] = np.max(vor_DNS_t)
            minMaxUVP[kk-4,5] = np.min(vor_DNS_t)

            filename = "results/plots_org/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
            print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, res, filename)

            filename = "results/fields_org/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
            save_fields(0, den_DNS_t, phi_DNS_t, vor_DNS_t, zero_DNS, zero_DNS, zero_DNS, filename)

            filename = "results/energy_org/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
            if (kk==RES_LOG2):
                closePlot=True
            plot_spectrum(den_DNS_t, phi_DNS_t, L, filename, close=closePlot)

            print("DNS spectrum at resolution " + str(res))

        os.system("mv Energy_spectrum.png results/energy_org/Energy_spectrum_org.png")


        # prepare latent space
        if (USE_DLATENTS):
            zlatent = tf.random.uniform([1, LATENT_SIZE])
            latent  = mapping(zlatent, training=False)
        else:
            latent = tf.random.uniform([1, LATENT_SIZE])
    

        # preprare target image
        kk = RES_LOG2
        res = 2**kk
        tminMaxUVP = tf.convert_to_tensor(minMaxUVP[RES_LOG2-4,:][np.newaxis,:], dtype="float32")
        s = int(res/DIM_DATA)
        rs = int(DIM_DATA/res)
        if (s==1):
            den_DNS_t = den_DNS[:,:]
            phi_DNS_t = phi_DNS[:,:]
            vor_DNS_t = vor_DNS[:,:]
        else:            
            den_DNS_t = sc.ndimage.gaussiaden_filter(den_DNS, rs, mode='grid-wrap')
            phi_DNS_t = sc.ndimage.gaussiaden_filter(phi_DNS, rs, mode='grid-wrap')
            vor_DNS_t = sc.ndimage.gaussiaden_filter(vor_DNS, rs, mode='grid-wrap')

            den_DNS_t = den_DNS_t[::rs, ::rs]
            phi_DNS_t = phi_DNS_t[::rs, ::rs]
            vor_DNS_t = vor_DNS_t[::rs, ::rs]

        tden_DNS = tf.convert_to_tensor(den_DNS_t)
        tphi_DNS = tf.convert_to_tensor(phi_DNS_t)
        tvor_DNS = tf.convert_to_tensor(vor_DNS_t)

        den_DNS = tden_DNS[np.newaxis,np.newaxis,:,:]
        phi_DNS = tphi_DNS[np.newaxis,np.newaxis,:,:]
        vor_DNS = tvor_DNS[np.newaxis,np.newaxis,:,:]
        imgA = tf.concat([den_DNS, phi_DNS, vor_DNS], 1)


        # optimize latent space
        itDNS  = 0
        resDNS = large
        tstart = time.time()
        while (resDNS>tollDNS and itDNS<maxItDNS):
            resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step(latent, tminMaxUVP, imgA, list_DNS_trainable_variables)

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
                den_DNS_t = UVP_DNS[0, 0, :, :].numpy()
                phi_DNS_t = UVP_DNS[0, 1, :, :].numpy()
                vor_DNS_t = UVP_DNS[0, 2, :, :].numpy()


                filename = "results/plots/Plots_DNS_fromGAN.png"
                #filename = "results/plots/Plots_DNS_fromGAN" + str(itDNS) + ".png"

                #print_fields_1(W_DNS_t, filename)
                print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, N_DNS, filename)

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
        resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step(latent, tminMaxUVP, imgA, list_DNS_trainable_variables)

        den_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        phi_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        vor_DNS_t = UVP_DNS[0, 2, :, :].numpy()

        print_fields_1(vor_DNS_t,     "results/plots/Plots_DNS_fromGAN.png", legend=False)
        print_fields_1(vor_DNS_org,   "results/plots_org/Plots_DNS_org.png", legend=False)

        # save checkpoint for wl_synthesis
        wl_checkpoint.save(file_prefix = WL_CHKP_PREFIX)

    else:

        # find DNS and LES fields from random input
        tminMaxUVP = tf.convert_to_tensor(minMaxUVP[RES_LOG2-4,:][np.newaxis,:], dtype="float32")
        if (USE_DLATENTS):
            zlatent              = tf.random.uniform([1, LATENT_SIZE])
            dlatents             = mapping(zlatent, training=False)
            predictions, UVP_DNS = wl_synthesis([dlatents, tminMaxUVP], training=False)
        else:
            latents               = tf.random.uniform([1, LATENT_SIZE])
            predictions, UVP_DNS  = wl_synthesis([latents, tminMaxUVP], training=False)



    # print spectrum from filter
    UVP_LES = filter(UVP_DNS, training=False)
    res = 2**RES_LOG2_FIL
    den_t = UVP_LES[0, 0, :, :].numpy()
    phi_t = UVP_LES[0, 1, :, :].numpy()
    vor_t = UVP_LES[0, 2, :, :].numpy()

    filename = "results/plots/plots_fil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(den_t, phi_t, vor_t, res, filename)

    filename = "results/fields/fields_fil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0, den_t, phi_t, vor_t, zero_LES, zero_LES, zero_LES, filename)

    filename = "results/energy/energy_spectrum_fil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(den_t, phi_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_filtered.png")




    # write fields and energy spectra for each layer
    closePlot=False
    for kk in range(4, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]
        res = 2**kk

        den_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        phi_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        vor_DNS_t = UVP_DNS[0, 2, :, :].numpy()
        
        den_DNS_t = two*(den_DNS_t - np.min(den_DNS_t))/(np.max(den_DNS_t) - np.min(den_DNS_t)) - one
        phi_DNS_t = two*(phi_DNS_t - np.min(phi_DNS_t))/(np.max(phi_DNS_t) - np.min(phi_DNS_t)) - one
        vor_DNS_t = two*(vor_DNS_t - np.min(vor_DNS_t))/(np.max(vor_DNS_t) - np.min(vor_DNS_t)) - one

        den_DNS_t = (den_DNS_t+one)*(minMaxUVP[kk-4,0]-minMaxUVP[kk-4,1])/two + minMaxUVP[kk-4,1]
        phi_DNS_t = (phi_DNS_t+one)*(minMaxUVP[kk-4,2]-minMaxUVP[kk-4,3])/two + minMaxUVP[kk-4,3]
        vor_DNS_t = (vor_DNS_t+one)*(minMaxUVP[kk-4,4]-minMaxUVP[kk-4,5])/two + minMaxUVP[kk-4,5]

        filename = "results/plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, res, filename)

        filename = "results/fields/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
        save_fields(0, den_DNS_t, phi_DNS_t, vor_DNS_t, zero_DNS, zero_DNS, zero_DNS, filename)

        filename = "results/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
        if (kk==RES_LOG2):
            closePlot=True
        plot_spectrum(den_DNS_t, phi_DNS_t, L, filename, close=closePlot)

        print("From GAN spectrum at resolution " + str(res))


    print_fields_1(vor_DNS_t, "results/plots/Plots_DNS_fromGAN.png", legend=False)
    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_fromGAN.png")

    print ("done lantent " + str(k))
