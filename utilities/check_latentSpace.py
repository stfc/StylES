import os
import sys

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *
from HIT_2D import L, uRef

os.chdir('../')
from MSG_StyleGAN_tf2 import *
from IO_functions import StyleGAN_load_fields
os.chdir('./utilities')

from tensorflow.keras.applications.vgg16 import VGG16



# local parameters
CHECK      = "LATENTS"   # "LATENTS" consider also mapping, DLATENTS only synthetis
NL         = 10         # number of different latent vectors randomly selected
LOAD_FIELD = False       # load field from DNS solver (via restart.npz file)
FILE_REAL  = "../LES_Solvers/restart.npz"


# clean up
os.system("rm -rf plots")
os.system("rm -rf uvw")
os.system("rm -rf energy")
os.system("rm -rf log*")
os.system("mkdir plots")
os.system("mkdir uvw")
os.system("mkdir energy")

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
mapping_ave.trainable = False
synthesis_ave.trainable = False


# create variable synthesis model
if (CHECK=="DLATENTS"):
    dlatents     = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
    wlatents     = layer_wlatent(dlatents)
    ndlatents    = wlatents(dlatents)
    outputs      = synthesis_ave(ndlatents, training=False)
    wl_synthesis = tf.keras.Model(dlatents, outputs)
else:
    latents      = tf.keras.Input(shape=[LATENT_SIZE])
    wlatents     = layer_wlatent(latents)
    nlatents     = wlatents(latents)
    dlatents     = mapping_ave(nlatents)
    outputs      = synthesis_ave(dlatents, training=False)
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


# define step for finding latent space
@tf.function
def find_latent_step(latent, imgB):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(latent, training=False)
        UVW_DNS     = predictions[RES_LOG2-2]*2*uRef - uRef
        imgA        = UVW_DNS[:,:,:,:]
        loss_fea    = VGG_loss(imgA, imgB, VGG_extractor) 
        resDNS      = loss_fea[0]*iOUTDIM22 + loss_fea[1]    # loss pixel + sum loss features

        gradients_DNS  = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))

    return resDNS, predictions, UVW_DNS


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

        # normalize values
        orig[:,:,0:2] = (orig[:,:,0:2] - np.min(orig[:,:,0:2]))/(np.max(orig[:,:,0:2]) - np.min(orig[:,:,0:2]))
        orig[:,:,  2] = (orig[:,:,  2] - np.min(orig[:,:,  2]))/(np.max(orig[:,:,  2]) - np.min(orig[:,:,  2]))

        U_DNS = orig[:,:,0]
        V_DNS = orig[:,:,1]
        W_DNS = find_vorticity(U_DNS, V_DNS)

        print_fields_1(W_DNS, "Plots_DNS_org.png")
        W_DNS_org = W_DNS

        # start iteration search latent space
        itDNS        = 0
        resDNS       = large
        tstart = time.time()
        if (CHECK=="DLATENTS"):
            zlatent   = tf.random.uniform([1, LATENT_SIZE])
            newlatent = mapping_ave(zlatent, training=False)
        else:
            newlatent = tf.random.uniform([1, LATENT_SIZE])
    
        tU_DNS = tf.convert_to_tensor(U_DNS)
        tV_DNS = tf.convert_to_tensor(V_DNS)
        tW_DNS = tf.convert_to_tensor(W_DNS)

        U_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        V_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        W_DNS = tW_DNS[np.newaxis,np.newaxis,:,:]
        imgB = tf.concat([U_DNS, V_DNS, W_DNS], 1)

        # first seek best seed
        while (resDNS>tollDNS and itDNS<100):
            predictions = wl_synthesis(newlatent, training=False)
            UVW_DNS     = predictions[RES_LOG2-2]*2*uRef - uRef
            imgA        = UVW_DNS[:,:,:,:]
            loss_fea    = VGG_loss(imgA, imgB, VGG_extractor) 
            newResDNS   = loss_fea[0]*iOUTDIM22 + loss_fea[1]    # loss pixel + sum loss features

            if (newResDNS < resDNS):
                resDNS = newResDNS
                latent = newlatent

                # print the fields 
                tend = time.time()
                print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}".format(tend-tstart, itDNS, resDNS.numpy()))
                W_DNS_t = UV_DNS[0, 2, :, :].numpy()

                print_fields_1(W_DNS_t, "Plots_DNS_fromGAN.png")
            else:
                if (CHECK=="DLATENTS"):
                    zlatent   = tf.random.uniform([1, LATENT_SIZE])
                    newlatent = mapping_ave(zlatent, training=False)
                else:
                    newlatent = tf.random.uniform([1, LATENT_SIZE])

            itDNS = itDNS+1


        # now optimize latent space
        itDNS        = 0
        resDNS       = large
        while (resDNS>tollDNS and itDNS<maxItDNS):
            resDNS, predictions, UVW_DNS = find_latent_step(latent, imgB)

            # write values in tensorboard
            lr = lr_schedule(itDNS)
            with train_summary_writer.as_default():
                tf.summary.scalar("residuals", resDNS, step=itDNS)
                tf.summary.scalar("lr", lr, step=itDNS)

            # print the fields 
            if (itDNS%1000 == 0):
                tend = time.time()
                print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  lr {3:3e} ".format(tend-tstart, itDNS, resDNS.numpy(), lr))
                U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
                V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
                W_DNS_t = UVW_DNS[0, 2, :, :].numpy()

                print_fields_1(W_DNS_t, "Plots_DNS_fromGAN.png")

            itDNS = itDNS+1

        # reprint but without legends
        print_fields_1(W_DNS_t, "Plots_DNS_fromGAN.png", legend=False)
        print_fields_1(W_DNS_org,   "Plots_DNS_org.png",     legend=False)

    else:

        # find DNS and LES fields from random input 
        if (CHECK=="DLATENTS"):
            zlatent     = tf.random.uniform([1, LATENT_SIZE])
            dlatents    = mapping_ave(zlatent, training=False)
            predictions = wl_synthesis(dlatents, training=False)
        else:
            latents      = tf.random.uniform([1, LATENT_SIZE])
            predictions  = wl_synthesis(latents, training=False)



    # write fields and energy spectra
    closePlot=False
    for kk in range(RES_LOG2-2, RES_LOG2-1):
        UVW_DNS = predictions[kk]*2*uRef - uRef
        res = 2**(kk+2)
        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
        W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
        #W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        filename = "plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields(U_DNS_t, V_DNS_t, U_DNS_t, W_DNS_t, res, filename)

        filename = "energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
        if (kk== RES_LOG2-2):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

    print ("done lantent " + str(k))
