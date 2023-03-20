import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.ticker import FormatStrFormatter

print("TensorFlow version:", tf.__version__)




filename = "../../../../results/HIT_2D/fREC1000/uvw_vs_time.npz"
#filename = "../../../../results/uvw_vs_time_mHW_fREC100.npz"
data = np.load(filename)

totTime=data['totTime']
velx_DNS=data['U_DNS']
vely_DNS=data['V_DNS']
vort_DNS=data['W_DNS']

mP = int(velx_DNS.shape[2]/1)

fig, axs = plt.subplots(1, 3, figsize=(20,10))
fig.subplots_adjust(hspace=0.25)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
colors = ['k','r','b','g','y']

tollLESValues = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]

# start main loop
for tv, tollLES in enumerate(tollLESValues):

    lineColor = colors[tv]
    stollLES = "{:.1e}".format(tollLES)

    if (tv==0):
        ax1.plot(totTime[:mP], velx_DNS[tv,0,:mP], color=lineColor, label=r'DNS $u$')
        ax2.plot(totTime[:mP], vely_DNS[tv,0,:mP], color=lineColor, label=r'DNS $v$')
        ax3.plot(totTime[:mP], vort_DNS[tv,0,:mP], color=lineColor, label=r'DNS $\zeta$')

    lineColor = colors[tv+1]
    stollLES = "{:.1e}".format(tollLES)

    if (tv>=0):
        ax1.plot(totTime[:mP], velx_DNS[tv,1,:mP], color=lineColor, linestyle='dashed', label=r'StylES DNS $n$ at toll ' + stollLES)
        ax2.plot(totTime[:mP], vely_DNS[tv,1,:mP], color=lineColor, linestyle='dashed', label=r'StylES DNS $\phi$ at toll ' + stollLES)
        ax3.plot(totTime[:mP], vort_DNS[tv,1,:mP], color=lineColor, linestyle='dashed', label=r'StylES DNS $\zeta$ at toll ' + stollLES)


ax1.legend()
ax2.legend()
ax3.legend()

plt.savefig('reconstructions_fREC1.png')

