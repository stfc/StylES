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
USE_DLATENTS   = "DLATENTS"   # "LATENTS" consider also mapping, DLATENTS only synthesis
NL             = 1         # number of different latent vectors randomly selected
LOAD_FIELD     = True       # load field from DNS solver (via restart.npz file)
RES_TARGET     = RES_LOG2   # target for the matching fields. Usually equal to the filter style
FILE_REAL      = "../../../data/N256_test_procedures/fields/fields_run0_it100.npz"
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
tf.random.set_seed(2)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  

N_DNS = 2**RES_LOG2
N_LES = 2**RES_LOG2_FIL
C_DNS = cp.zeros([N_DNS,N_DNS], dtype=DTYPE)
B_DNS = cp.zeros([N_DNS,N_DNS], dtype=DTYPE)
C_LES = cp.zeros([N_LES, N_LES], dtype=DTYPE)
B_LES = cp.zeros([N_LES, N_LES], dtype=DTYPE)


with mirrored_strategy.scope():
        
    # define noise variances
    inputVar1 = tf.constant(1.0, shape=[BATCH_SIZE, G_LAYERS-2], dtype=DTYPE)
    inputVar2 = tf.constant(1.0, shape=[BATCH_SIZE, 2], dtype=DTYPE)
    inputVariances = tf.concat([inputVar1,inputVar2],1)


    # Download VGG16 model
    VGG_model         = VGG16(input_shape=(OUTPUT_DIM, OUTPUT_DIM, NUM_CHANNELS), include_top=False, weights='imagenet')
    VGG_features_list = [layer.output for layer in VGG_model.layers]
    VGG_extractor     = tf.keras.Model(inputs=VGG_model.input, outputs=VGG_features_list)


    # loading StyleGAN checkpoint and filter
    checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))


    # create variable synthesis model
    if (USE_DLATENTS=="DLATENTS"):
        dlatents       = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
        wlatents       = layer_wlatent(dlatents)
        ndlatents      = wlatents(dlatents)
        outputs        = synthesis([ndlatents, inputVariances], training=False)
        wl_synthesis   = tf.keras.Model(dlatents, outputs)
    else:
        latents        = tf.keras.Input(shape=[LATENT_SIZE])
        wlatents       = layer_wlatent(latents)
        nlatents       = wlatents(latents)
        dlatents       = mapping(nlatents)
        outputs        = synthesis([dlatents, inputVariances], training=False)
        wl_synthesis   = tf.keras.Model(latents, outputs)


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
    wl_checkpoint = tf.train.Checkpoint(wl_synthesis=wl_synthesis,
                                        opt=opt)


    # Load latest checkpoint, if restarting
    if (WL_IRESTART):
        status = wl_checkpoint.restore(tf.train.latest_checkpoint(WL_CHKP_DIR))




# add latent space to trainable variables
for variable in wlatents.trainable_variables:
    list_LES_trainable_variables.append(variable)




# define step for finding latent space
@tf.function
def find_latent(latent, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(latent, training=False)
        UVP_DNS     = predictions[RES_LOG2-2]
        UVP_LES     = predictions[RES_TARGET-2]
        #UVP_LES    = filter(UVP_DNS, training=False)
        new_img     = UVP_LES[:,:,:,:]

        U = new_img[0,0,:,:]
        V = new_img[0,1,:,:]
        P = new_img[0,2,:,:]
        #W = ((tr(V, 1, 0)-tr(V, -1, 0)) - (tr(U, 0, 1)-tr(U, 0, -1)))

        U = U[np.newaxis, np.newaxis, :, :]
        V = V[np.newaxis, np.newaxis, :, :]
        P = P[np.newaxis, np.newaxis, :, :]
        #W = W[np.newaxis, np.newaxis, :, :]

        imgB = tf.concat([U,V,P], 1)
        #imgB = tf.concat([U,V,W], 1)

        # normalize values
        # loss_fea = VGG_loss(imgA, imgB, VGG_extractor) 
        # resDNS  =  tf.math.reduce_sum(loss_fea[2:])
        loss_pix = tf.math.reduce_mean(tf.math.squared_difference(imgA[0,:,:,:], imgB[0,:,:,:]))
        resDNS   = tf.math.reduce_sum(loss_pix)

        gradients_DNS  = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

    return resDNS, predictions, UVP_DNS, loss_pix


@tf.function
def find_latent_step(input, images, list_trainable_variables):
    resDNS, predictions, UVP_DNS, loss_fea = mirrored_strategy.run(find_latent, args=(input, images, list_trainable_variables))
    return resDNS, predictions, UVP_DNS, loss_fea



# print different fields (to check quality and find 2 different seeds)
for k in range(NL):
    
    # load initial flow
    tf.random.set_seed(k)
    if (LOAD_FIELD):

        if (FILE_REAL.endswith('.npz')):
            
            # load numpy array
            orig = np.zeros([OUTPUT_DIM,OUTPUT_DIM, 3], dtype=DTYPE)
            img_in = StyleGAN_load_fields(FILE_REAL)
            orig[:,:,0] = img_in[-1][0,:,:]
            orig[:,:,1] = img_in[-1][1,:,:]
            orig[:,:,2] = img_in[-1][2,:,:]
            orig = np.cast[DTYPE](orig)

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

        # normalize values
        U_DNS = orig[:,:,0]
        V_DNS = orig[:,:,1]
        P_DNS = orig[:,:,2]
        W_DNS = find_vorticity(U_DNS, V_DNS)

        #print_fields_1(W_DNS, "Plots_DNS_org.png")
        print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, "results/plots_org/Plots_DNS_org.png")
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
            else:
                # U_DNS_t = sc.ndimage.gaussian_filter(U_DNS, rs, mode='grid-wrap')
                # V_DNS_t = sc.ndimage.gaussian_filter(V_DNS, rs, mode='grid-wrap')
                # P_DNS_t = sc.ndimage.gaussian_filter(P_DNS, rs, mode='grid-wrap')

                # U_DNS_t = U_DNS_t[::rs, ::rs]
                # V_DNS_t = V_DNS_t[::rs, ::rs]
                # P_DNS_t = P_DNS_t[::rs, ::rs]

                # preprare Gaussian Kernel
                gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
                gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

                # convert to tensor
                U_DNS_t = tf.convert_to_tensor(U_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                V_DNS_t = tf.convert_to_tensor(V_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)
                P_DNS_t = tf.convert_to_tensor(P_DNS[np.newaxis,:,:,np.newaxis], dtype=DTYPE)

                # add padding
                pleft   = 4*rs
                pright  = 4*rs
                ptop    = 4*rs
                pbottom = 4*rs

                U_DNS_t = periodic_padding_flexible(U_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                V_DNS_t = periodic_padding_flexible(V_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
                P_DNS_t = periodic_padding_flexible(P_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

                # convolve
                U_DNS_t = tf.nn.conv2d(U_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                V_DNS_t = tf.nn.conv2d(V_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
                P_DNS_t = tf.nn.conv2d(P_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

                # downscale
                U_DNS_t = U_DNS_t[0,::rs,::rs,0].numpy()
                V_DNS_t = V_DNS_t[0,::rs,::rs,0].numpy()
                P_DNS_t = P_DNS_t[0,::rs,::rs,0].numpy()

            W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

            filename = "results/plots_org/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
            print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, res, filename)

            filename = "results/fields_org/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
            save_fields(0, U_DNS_t, V_DNS_t, P_DNS_t, C_DNS, B_DNS, W_DNS_t, filename)

            filename = "results/energy_org/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
            if (kk==RES_LOG2):
                closePlot=True
            plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

            print("DNS spectrum at resolution " + str(res))

        os.system("mv Energy_spectrum.png results/energy_org/Energy_spectrum_org.png")


        # prepare latent space
        if (USE_DLATENTS=="DLATENTS"):
            zlatent = tf.random.uniform([1, LATENT_SIZE])
            latent  = mapping(zlatent, training=False)
        else:
            latent = tf.random.uniform([1, LATENT_SIZE])
    

        # preprare target image
        kk = RES_TARGET
        res = 2**kk
        s = int(res/DIM_DATA)
        rs = int(DIM_DATA/res)
        if (s==1):
            U_DNS_t = U_DNS[:,:]
            V_DNS_t = V_DNS[:,:]
            P_DNS_t = P_DNS[:,:]
        else:            
            U_DNS_t = sc.ndimage.gaussian_filter(U_DNS, rs, mode='grid-wrap')
            V_DNS_t = sc.ndimage.gaussian_filter(V_DNS, rs, mode='grid-wrap')
            P_DNS_t = sc.ndimage.gaussian_filter(P_DNS, rs, mode='grid-wrap')

            U_DNS_t = U_DNS_t[::rs, ::rs]
            V_DNS_t = V_DNS_t[::rs, ::rs]
            P_DNS_t = P_DNS_t[::rs, ::rs]

        W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)            

        tU_DNS = tf.convert_to_tensor(U_DNS_t)
        tV_DNS = tf.convert_to_tensor(V_DNS_t)
        tP_DNS = tf.convert_to_tensor(P_DNS_t)
        tW_DNS = tf.convert_to_tensor(W_DNS_t)

        U_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        V_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        P_DNS = tP_DNS[np.newaxis,np.newaxis,:,:]
        W_DNS = tW_DNS[np.newaxis,np.newaxis,:,:]

        imgA = tf.concat([U_DNS, V_DNS, P_DNS], 1)
        #imgA = tf.concat([U_DNS, V_DNS, W_DNS], 1)


        # optimize latent space
        itDNS  = 0
        resDNS = large
        tstart = time.time()
        while (resDNS>tollDNS and itDNS<maxItDNS):
            resDNS, predictions, UVP_DNS, loss_UV = find_latent_step(latent, imgA, list_LES_trainable_variables)

            # print residuals and fields
            if (itDNS%1000 == 0):

                # find learning rate
                lr = lr_schedule(itDNS)
                with train_summary_writer.as_default():
                    tf.summary.scalar("residuals/loss", resDNS, step=itDNS)
                    tf.summary.scalar("residuals/pixel_loss_UV", loss_UV, step=itDNS)
                    tf.summary.scalar("lr", lr, step=itDNS)

                # print residuals
                tend = time.time()
                print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  pixel loss UV {3:3e}  lr {4:3e} ".format(tend-tstart, itDNS, resDNS.numpy(), loss_UV, lr))

                # # print fields
                # U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
                # V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
                # P_DNS_t = UVP_DNS[0, 2, :, :].numpy()
                # W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

                # filename = "results/plots/Plots_DNS_fromGAN.png"
                # #filename = "results/plots/Plots_DNS_fromGAN" + str(itDNS) + ".png"

                # #print_fields_1(W_DNS_t, filename)
                # print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, N, filename)
               
            itDNS = itDNS+1

        # print final values
        lr = lr_schedule(itDNS)
        with train_summary_writer.as_default():
            tf.summary.scalar("residuals/loss", resDNS, step=itDNS)
            tf.summary.scalar("residuals/pixel_loss_UV", loss_UV, step=itDNS)
            tf.summary.scalar("lr", lr, step=itDNS)

        # print residuals
        tend = time.time()
        print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  pixel loss UV {3:3e}  lr {4:3e} ".format(tend-tstart, itDNS, resDNS.numpy(), loss_UV, lr))

        # reprint but only vorticity
        resDNS, predictions, UVP_DNS, _ = find_latent_step(latent, imgA, list_LES_trainable_variables)

        U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        P_DNS_t = UVP_DNS[0, 2, :, :].numpy()
        W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        print_fields_1(W_DNS_t, "results/plots/Plots_DNS_fromGAN.png", legend=False)
        print_fields_1(W_DNS_org,   "results/plots_org/Plots_DNS_org.png", legend=False)

        # save checkpoint for wl_synthesis
        wl_checkpoint.save(file_prefix = WL_CHKP_PREFIX)

    else:

        # find DNS and LES fields from random input 
        if (USE_DLATENTS=="DLATENTS"):
            zlatent     = tf.random.uniform([1, LATENT_SIZE])
            dlatents    = mapping(zlatent, training=False)
            predictions = wl_synthesis([dlatents, inputVariances], training=False)
        else:
            latents      = tf.random.uniform([1, LATENT_SIZE])
            predictions  = wl_synthesis([latents, inputVariances], training=False)




    # print spectrum from filter
    UVP_DNS = predictions[RES_LOG2-2]
    UVP_LES = filter(UVP_DNS, training=False)
    res = 2**RES_LOG2_FIL
    U_t = UVP_LES[0, 0, :, :].numpy()
    V_t = UVP_LES[0, 1, :, :].numpy()
    P_t = UVP_LES[0, 2, :, :].numpy()
    W_t = find_vorticity(U_t, V_t)

    filename = "results/plots/plots_fil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields(U_t, V_t, P_t, W_t, res, filename)

    filename = "results/fields/fields_fil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0, U_t, V_t, P_t, C_LES, B_LES, W_t, filename)

    filename = "results/energy/energy_spectrum_fil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(U_t, V_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_filtered.png")




    # write fields and energy spectra for each layer
    closePlot=False
    for kk in range(4, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]
        res = 2**kk

        U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        P_DNS_t = UVP_DNS[0, 2, :, :].numpy()
        W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        filename = "results/plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, res, filename)

        filename = "results/fields/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
        save_fields(0, U_DNS_t, V_DNS_t, P_DNS_t, C_DNS, B_DNS, W_DNS_t, filename)

        filename = "results/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
        if (kk==RES_LOG2):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

        print("From GAN spectrum at resolution " + str(res))


    os.system("mv Energy_spectrum.png results/energy/Energy_spectrum_fromGAN.png")

    print ("done lantent " + str(k))
