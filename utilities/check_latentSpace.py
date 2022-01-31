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
os.chdir('./utilities')

from tensorflow.keras.applications.vgg16 import VGG16



# local parameters
CHECK       = "DLATENTS"   # "LATENTS" consider also mapping, DLATENTS only synthetis
NL          = 1         # number of different latent vectors randomly selected
LOAD_FIELD  = False       # load field from DNS solver (via restart.npz file)
FILE_REAL   = "../../../data/N1024_single/fields/fields_run0_134te.npz"
WL_IRESTART = False
WL_CHKP_DIR = './wl_checkpoints/'


# clean up
if LOAD_FIELD:
    if TRAIN:
        print("Set TRAIN flag to False in parameters!")
        exit()    


os.system("rm -rf plots")
os.system("rm -rf fields")
os.system("rm -rf uvw")
os.system("rm -rf energy")
os.system("rm -rf logs")

os.system("mkdir plots")
os.system("mkdir fields")
os.system("mkdir uvw")
os.system("mkdir energy")
os.system("mkdir logs")

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(2)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  


# Download VGG16 model
VGG_model         = VGG16(input_shape=(OUTPUT_DIM, OUTPUT_DIM, NUM_CHANNELS), include_top=False, weights='imagenet')
VGG_features_list = [layer.output for layer in VGG_model.layers]
VGG_extractor     = tf.keras.Model(inputs=VGG_model.input, outputs=VGG_features_list)


# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))


# create variable synthesis model
if (CHECK=="DLATENTS"):
    dlatents     = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
    wlatents     = layer_wlatent(dlatents)
    ndlatents    = wlatents(dlatents)
    outputs      = synthesis(ndlatents, training=False)
    wl_synthesis = tf.keras.Model(dlatents, outputs)
else:
    latents      = tf.keras.Input(shape=[LATENT_SIZE])
    wlatents     = layer_wlatent(latents)
    nlatents     = wlatents(latents)
    dlatents     = mapping(nlatents)
    outputs      = synthesis(dlatents, training=False)
    wl_synthesis = tf.keras.Model(latents, outputs)


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
checkpoint = tf.train.Checkpoint(wl_synthesis=wl_synthesis,
                                 opt=opt)



# define step for finding latent space
@tf.function
def find_latent_step(latent, imgA):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(latent, training=False)
        UVW_DNS     = predictions[RES_LOG2-2]
        new_img     = UVW_DNS[:,:,:,:]

        U = new_img[0,0,:,:]
        V = new_img[0,1,:,:]
        W = ((tr(V, 1, 0)-tr(V, -1, 0)) - (tr(U, 0, 1)-tr(U, 0, -1)))

        U = U[np.newaxis, np.newaxis, :, :]
        V = V[np.newaxis, np.newaxis, :, :]
        W = W[np.newaxis, np.newaxis, :, :]

        imgB = tf.concat([U,V,W], 1)

        # normalize values
        loss_fea    = VGG_loss(imgA, imgB, VGG_extractor) 
        resDNS      =  tf.math.reduce_sum(loss_fea[2:])    # loss pixel (UV) + sum loss features

        gradients_DNS  = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))

    return resDNS, predictions, UVW_DNS, loss_fea


# Load latest checkpoint, if restarting
if (WL_IRESTART):
    checkpoint.restore(tf.train.latest_checkpoint(WL_CHKP_DIR))


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
        W_DNS = find_vorticity(U_DNS, V_DNS)

        #print_fields_1(W_DNS, "Plots_DNS_org.png")
        print_fields(U_DNS, V_DNS, U_DNS, W_DNS, N, "Plots_DNS_org.png")
        W_DNS_org = W_DNS


        # write fields and energy spectra of DNS
        DIM_DATA, _ = U_DNS.shape
        closePlot=False
        for kk in range(0, RES_LOG2-3):
            res = 2**(kk+4)
            s = int(res/DIM_DATA)
            rs = int(DIM_DATA/res)
            if (s == 1):
                U_DNS_t = U_DNS[:,:]
                V_DNS_t = V_DNS[:,:]
            else:
                U_DNS_t = sc.ndimage.gaussian_filter(U_DNS, rs*np.sqrt(1.0/12.0), mode='grid-wrap')
                V_DNS_t = sc.ndimage.gaussian_filter(V_DNS, rs*np.sqrt(1.0/12.0), mode='wrap')

                U_DNS_t = U_DNS_t[::rs, ::rs]
                V_DNS_t = V_DNS_t[::rs, ::rs]

            W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

            filename = "plots/plots_org_lat_" + str(k) + "_res_" + str(res) + ".png"
            print_fields(U_DNS_t, V_DNS_t, U_DNS_t, W_DNS_t, res, filename)

            filename = "fields/fields_org_lat_" + str(k) + "_res_" + str(res) + ".npz"
            save_fields(0, U_DNS_t, V_DNS_t, U_DNS_t, U_DNS_t, U_DNS_t, W_DNS_t, filename)

            filename = "energy/energy_org_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
            if (kk== RES_LOG2-4):
                closePlot=True
            plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

            print("DNS spectrum at resolution " + str(res))

        os.system("mv Energy_spectrum.png Energy_spectrum_org.png")

        # prepare latent space
        if (CHECK=="DLATENTS"):
            zlatent = tf.random.uniform([1, LATENT_SIZE])
            latent  = mapping(zlatent, training=False)
        else:
            latent = tf.random.uniform([1, LATENT_SIZE])
    

        # preprare reference image
        tU_DNS = tf.convert_to_tensor(U_DNS_t)
        tV_DNS = tf.convert_to_tensor(V_DNS_t)
        tW_DNS = tf.convert_to_tensor(W_DNS_t)
        U_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        V_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        W_DNS = tW_DNS[np.newaxis,np.newaxis,:,:]
        imgA = tf.concat([U_DNS, V_DNS, W_DNS], 1)


        # optimize latent space
        itDNS  = 0
        resDNS = large
        tstart = time.time()
        while (resDNS>tollDNS and itDNS<maxItDNS):
            resDNS, predictions, UVW_DNS, losses = find_latent_step(latent, imgA)

            # print the fields 
            if (itDNS%100 == 0):

                # find learning rate
                lr = lr_schedule(itDNS)
                with train_summary_writer.as_default():
                    tf.summary.scalar("residuals/loss", resDNS, step=itDNS)
                    tf.summary.scalar("residuals/pixel_loss_UV", losses[0], step=itDNS)
                    tf.summary.scalar("lr", lr, step=itDNS)

                # print residuals
                tend = time.time()
                print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  pixel loss UV {3:3e}  lr {4:3e} ".format(tend-tstart, itDNS, resDNS.numpy(), losses[1], lr))

                # print fields
                U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
                V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
                #W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
                W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

                filename = "Plots_DNS_fromGAN.png"
                #filename = "Plots_DNS_fromGAN" + str(itDNS) + ".png"

                #print_fields_1(W_DNS_t, filename)
                print_fields(U_DNS_t, V_DNS_t, U_DNS_t, W_DNS_t, N, filename)

            itDNS = itDNS+1

        # reprint but only vorticity
        resDNS, predictions, UVW_DNS, _ = find_latent_step(latent, imgA)

        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
        #W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
        W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        print_fields_1(W_DNS_t, "Plots_DNS_fromGAN.png", legend=False)
        print_fields_1(W_DNS_org,   "Plots_DNS_org.png", legend=False)

    else:

        # find DNS and LES fields from random input 
        if (CHECK=="DLATENTS"):
            zlatent     = tf.random.uniform([1, LATENT_SIZE])
            dlatents    = mapping(zlatent, training=False)
            predictions = wl_synthesis(dlatents, training=False)
        else:
            latents      = tf.random.uniform([1, LATENT_SIZE])
            predictions  = wl_synthesis(latents, training=False)



    # save checkpoint for wl_synthesis
    checkpoint.save(file_prefix = WL_CHKP_DIR)



    # print spectrum from filter
    UVW_DNS = predictions[RES_LOG2-2]
    UVW = filter(UVW_DNS, training=False)
    res = 2**RES_LOG2_FIL
    U_t = UVW[0, 0, :, :].numpy()  #*(maxList[kk]-minList[kk]) + minList[kk]
    V_t = UVW[0, 1, :, :].numpy()  #*(maxList[kk]-minList[kk]) + minList[kk]
    #W_t = UVW[0, 2, :, :].numpy()
    W_t = find_vorticity(U_t, V_t)

    filename = "plots/plots_fil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields(U_t, V_t, U_t, W_t, res, filename)

    filename = "fields/fields_fil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0, U_t, V_t, U_t, U_t, U_t, W_t, filename)

    filename = "energy/energy_spectrum_fil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(U_t, V_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png Energy_spectrum_filtered.png")




    # write fields and energy spectra for each layer
    closePlot=False
    for kk in range(0, RES_LOG2-3):
        UVW_DNS = predictions[kk+2]
        res = 2**(kk+4)

        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()

        # s = res/DIM_DATA
        # U_DNS_t = sc.ndimage.interpolation.zoom(UVW_DNS[0, 0, :, :].numpy(), s, order=3, mode='wrap')
        # V_DNS_t = sc.ndimage.interpolation.zoom(UVW_DNS[0, 1, :, :].numpy(), s, order=3, mode='wrap')

        #W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
        W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        filename = "plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields(U_DNS_t, V_DNS_t, U_DNS_t, W_DNS_t, res, filename)

        filename = "fields/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
        save_fields(0, U_DNS_t, V_DNS_t, U_DNS_t, U_DNS_t, U_DNS_t, W_DNS_t, filename)

        filename = "energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
        if (kk== RES_LOG2-4):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

        print("done energy spectrum for resolution " + str(res))


    print ("done lantent " + str(k))
