import os
import sys
import scipy as sc
import matplotlib.pyplot as plt

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from matplotlib.ticker import FormatStrFormatter
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
NL             = 1400       # number of different latent vectors randomly selected
TUNE_NOISE     = True
LOAD_FIELD     = True       # load field from DNS solver (via restart.npz file)
CALC_VORTICITY = True
USE_VGG        = False
FILE_REAL_PATH = "../LES_Solvers/fields/"
WL_IRESTART    = False
WL_CHKP_DIR    = "./wl_checkpoints"
WL_CHKP_PREFIX = os.path.join(WL_CHKP_DIR, "ckpt")

# clean up and prepare folders
os.system("rm -rf results_reconstruction/plots")
os.system("rm -rf results_reconstruction/fields")
os.system("rm -rf results_reconstruction/uvw")
os.system("rm -rf results_reconstruction/energy")
if (LOAD_FIELD):
    os.system("rm -rf results_reconstruction/plots_org")
    os.system("rm -rf results_reconstruction/fields_org")
    os.system("rm -rf results_reconstruction/uvw_org")
    os.system("rm -rf results_reconstruction/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results_reconstruction/plots")
os.system("mkdir -p results_reconstruction/fields")
os.system("mkdir -p results_reconstruction/uvw")
os.system("mkdir -p results_reconstruction/energy")
os.system("mkdir -p results_reconstruction/plots_org/")
os.system("mkdir -p results_reconstruction/fields_org")
os.system("mkdir -p results_reconstruction/uvw_org")
os.system("mkdir -p results_reconstruction/energy_org")

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(0)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  

N_DNS = 2**RES_LOG2
N_LES = 2**RES_LOG2_FIL
C_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
B_DNS = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
C_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
B_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)
nimgB = np.zeros([1, 3, N_DNS, N_DNS], dtype=DTYPE)
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
    if (lrREC_POLICY=="EXPONENTIAL"):
        lr_schedule  = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lrREC,
            decay_steps=lrREC_STEP,
            decay_rate=lrREC_RATE,
            staircase=lrREC_EXP_ST)
    elif (lrREC_POLICY=="PIECEWISE"):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrREC_BOUNDS, lrREC_VALUES)
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)




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




# define step for finding latent space
@tf.function
def find_latent_LES(latent, minMaxUVP, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions, UVP_DNS = wl_synthesis([latent, minMaxUVP], training=False)
        UVP_LES     = predictions[RES_LOG2_FIL-2]

        fimg = filter(imgA)
        fUVP_LES = filter(UVP_DNS)
        # fimg = filter(imgA)[0][2]  #  to use if filter spits back every layer. See commented line in MSG_StyleGAN file
        # fUVP_LES = filter(UVP_DNS)[0][2]


        # find loss
        # U = UVP_DNS[0,0,:,:]
        # V = UVP_DNS[0,1,:,:]
        # loss_div = tf.math.reduce_mean(tf.abs(((tr(U, 1, 0)-tr(U, -1, 0)) + (tr(V, 0, 1)-tr(V, 0, -1)))))
        if (USE_VGG):
            loss_fea     = VGG_loss_LES(imgA, fimg, VGG_extractor_LES) 
            loss_pix_DNS = 0.0
            loss_pix_LES = 0.0
            resDNS       = tf.math.reduce_sum(loss_fea)
        else:
            loss_pix_DNS = tf.math.reduce_mean(tf.math.squared_difference(fUVP_LES[0,:,:,:], fimg[0,:,:,:]))
            loss_pix_DNS = loss_pix_DNS/tf.math.reduce_mean(fimg[0,:,:,:]**2)
            loss_pix_LES = tf.math.reduce_mean(tf.math.squared_difference(UVP_LES[0,:,:,:], fimg[0,:,:,:]))
            loss_pix_LES = loss_pix_LES/tf.math.reduce_mean(fimg[0,:,:,:]**2)
            resDNS       = loss_pix_DNS + loss_pix_LES

        gradients_DNS  = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

    return resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES, fimg


@tf.function
def find_latent_step_LES(latent, minMaxUVP, images, list_trainable_variables):
    resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES, fimg = mirrored_strategy.run(find_latent_LES, args=(latent, minMaxUVP, images, list_trainable_variables))
    return resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES, fimg






tollDNSValues = [1.0e-1, 1.0e-2, 1.0e-3]
ltoll = len(tollDNSValues)
totTime = np.zeros((NL), dtype="float32")
velx = np.zeros((ltoll,2,NL), dtype="float32")
vely = np.zeros((ltoll,2,NL), dtype="float32")
vort = np.zeros((ltoll,2,NL), dtype="float32")
N2 = int(N/2)

fig, axs = plt.subplots(1, 3, figsize=(20,10))
fig.subplots_adjust(hspace=0.25)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
colors = ['k','r','b','g']

tstart   = time.time()
for tv, tollDNS in enumerate(tollDNSValues):

    # reload weights
    with mirrored_strategy.scope():
        # loading StyleGAN checkpoint and filter
        checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))

    # set latent spaces
    tf.random.set_seed(0)
    if (USE_DLATENTS):
        zlatent = tf.random.uniform([1, LATENT_SIZE])
        latent  = mapping(zlatent, training=False)
    else:
        latent = tf.random.uniform([1, LATENT_SIZE])

    for k in range(NL):
        
        # load initial flow
        tail = "it" + str(int(k*10+6040))
        FILE_REAL = FILE_REAL_PATH + "fields_run0_" + tail + ".npz"
        
        #-------------------------------------------------------- RECONSTRUCT
        # load numpy array
        U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime[k] = load_fields(FILE_REAL)
        U_DNS = np.cast[DTYPE](U_DNS)
        V_DNS = np.cast[DTYPE](V_DNS)
        P_DNS = np.cast[DTYPE](P_DNS)

        W_DNS = find_vorticity(U_DNS, V_DNS)

        if (tv==-1):
            filename = "results_reconstruction/plots_org/Plots_DNS_org_" + tail +".png"
            print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N_DNS, filename)

        velx[tv,0,k] = U_DNS[N2, N2]
        vely[tv,0,k] = V_DNS[N2, N2]
        vort[tv,0,k] = W_DNS[N2, N2]


        # write fields and energy spectra of DNS
        kk = RES_LOG2
        minMaxUVP[kk-4,0] = np.max(U_DNS)
        minMaxUVP[kk-4,1] = np.min(U_DNS)
        minMaxUVP[kk-4,2] = np.max(V_DNS)
        minMaxUVP[kk-4,3] = np.min(V_DNS)
        minMaxUVP[kk-4,4] = np.max(P_DNS)
        minMaxUVP[kk-4,5] = np.min(P_DNS)



        # preprare target image
        tminMaxUVP = tf.convert_to_tensor(minMaxUVP[RES_LOG2-4,:][np.newaxis,:], dtype="float32")
        U_DNS_t = U_DNS[:,:]
        V_DNS_t = V_DNS[:,:]
        P_DNS_t = P_DNS[:,:]

        W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)            

        tU_DNS = tf.convert_to_tensor(U_DNS_t)
        tV_DNS = tf.convert_to_tensor(V_DNS_t)
        tP_DNS = tf.convert_to_tensor(P_DNS_t)
        tW_DNS = tf.convert_to_tensor(W_DNS_t)

        U_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        V_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        P_DNS = tP_DNS[np.newaxis,np.newaxis,:,:]
        W_DNS = tW_DNS[np.newaxis,np.newaxis,:,:]

        imgA = tf.concat([U_DNS, V_DNS, W_DNS], 1)


        itDNS    = 0
        itDNStot = 0
        resDNS   = large
        while (resDNS>tollDNS and itDNS<maxItDNS):
            if (k==0):
                resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step(latent, tminMaxUVP, imgA, list_DNS_trainable_variables)
            else:
                resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES, fimg = find_latent_step_LES(latent, tminMaxUVP, imgA, list_LES_trainable_variables)

            # print residuals and fields
            if (itDNS%1000==0):

                # print residuals
                lr = lr_schedule(itDNStot)
                tend = time.time()
                print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e}  lDNS {4:3e}  lLES {5:3e}  lr {6:3e} " \
                    .format(tend-tstart, k, itDNS, resDNS.numpy(), loss_pix_DNS, loss_pix_LES, lr))

            itDNS = itDNS+1
            itDNStot = itDNStot+1


           
        # small retrain to readjust the reconstruction coefficients
        if (k==-1):
            itDNS  = 0
            resDNS = large
            nimgB[0,:,:,:] = UVP_DNS[0,:,:,:].numpy()
            imgB = tf.convert_to_tensor(nimgB)
            while (resDNS>tollDNS and itDNS<maxItDNS):
                resDNS, predictions, UVP_DNS, loss_pix_DNS, loss_pix_LES = find_latent_step(latent, tminMaxUVP, imgB, list_DNS_trainable_variables)

                # print residuals and fields
                if (itDNS%1000==0):

                    # print residuals
                    lr = lr_schedule(itDNStot)
                    tend = time.time()
                    print("Fine DNS iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e}  lDNS {4:3e}  lLES {5:3e}  lr {6:3e} " \
                        .format(tend-tstart, k, itDNS, resDNS.numpy(), loss_pix_DNS, loss_pix_LES, lr))

                itDNS = itDNS+1
                itDNStot = itDNStot+1

            lr = lr_schedule(itDNStot)
            tend = time.time()
            print("Fine DNS iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e}  lDNS {4:3e}  lLES {5:3e}  lr {6:3e} " \
                .format(tend-tstart, k, itDNS, resDNS.numpy(), loss_pix_DNS, loss_pix_LES, lr))

        # print residuals
        else:
            # print residuals
            lr = lr_schedule(itDNStot)
            tend = time.time()
            print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e}  lDNS {4:3e}  lLES {5:3e}  lr {6:3e} " \
                .format(tend-tstart, k, itDNS, resDNS.numpy(), loss_pix_DNS, loss_pix_LES, lr))



        U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        W_DNS_t = UVP_DNS[0, 2, :, :].numpy()

        if (CALC_VORTICITY):
            W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

        if (tv==len(tollDNSValues)-1):
            filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + tail + "_" + str(tv) + ".png"
            print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, N_DNS, filename)

            filename = "results_reconstruction/plots/Vorticity_DNS_fromGAN_" + tail + "_" + str(tv) + ".png"
            print_fields_1(W_DNS_t, filename, Wmin = -0.3, Wmax = 0.3)

        N2 = int(N/2)
        velx[tv,1,k] = U_DNS_t[N2, N2]
        vely[tv,1,k] = V_DNS_t[N2, N2]
        vort[tv,1,k] = W_DNS_t[N2, N2]

    if (tv==0):
        lineColor = colors[tv]
        stollDNS = "{:.1e}".format(tollDNS)
        ax1.plot(totTime[:], velx[tv,0,:], color=lineColor, label='DNS x-vel')
        ax2.plot(totTime[:], vely[tv,0,:], color=lineColor, label='DNS y-vel')
        ax3.plot(totTime[:], vort[tv,0,:], color=lineColor, label='DNS vorticity')

    lineColor = colors[tv+1]
    stollDNS = "{:.1e}".format(tollDNS)
    ax1.plot(totTime[:], velx[tv,1,:], color=lineColor, linestyle='dashed', label='StylES x-vel at toll ' + stollDNS)
    ax2.plot(totTime[:], vely[tv,1,:], color=lineColor, linestyle='dashed', label='StylES y-vel at toll ' + stollDNS)
    ax3.plot(totTime[:], vort[tv,1,:], color=lineColor, linestyle='dashed', label='StylES vorticity at toll ' + stollDNS)

ax1.legend()
ax2.legend()
ax3.legend()

plt.savefig("uvw_vs_time.png")
plt.close()

np.savez("uvw_vs_time.npz", totTime=totTime, U=velx, V=vely, W=vort)





