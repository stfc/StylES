import numpy as np
import matplotlib.pyplot as plt


#------------------------------------ energy vs time
fig, ax1 = plt.subplots()

# # mHW 1024
# # ax1.set_yscale('log')
# color = 'tab:red'
# ax1.set_xlabel(r"time [$\omega^{-1}_i$]")
# ax1.set_ylabel(r"$\mathcal{F} (\phi)$", color=color)

# (ttime, Energy) = np.loadtxt("./mHW_N1024/energy_vs_time.txt",dtype="float64",unpack=True)
# plt.plot(ttime, Energy, label="mHW $1024^2$", color=color, linestyle='dotted')

# ax1.tick_params(axis='y', labelcolor=color)

# plt.legend(frameon=False, loc='lower center')



# HW 256 and 512
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_yscale('log')

color = 'tab:blue'
ax2.set_ylabel(r"$\mathcal{F} (\phi)$", color=color)

(ttime, Energy) = np.loadtxt("./HW_N256/energy_vs_time.txt",dtype="float64",unpack=True)
plt.plot(ttime, Energy, label="HW $256^2$", color=color, linestyle='solid')

(ttime, Energy) = np.loadtxt("./HW_N512/energy_vs_time.txt",dtype="float64",unpack=True)
plt.plot(ttime, Energy, label="HW $512^2$", color=color, linestyle='dashed' )

ax2.tick_params(axis='y', labelcolor=color)

plt.legend(frameon=False)


# print
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig('energy_vs_time.png')
plt.close()




#------------------------------------ spectra
plt.xscale("log")
plt.yscale("log")

(k, Energy) = np.loadtxt("./HW_N256/energy/Spectrum_1000.txt",dtype="float64",unpack=True)
plt.plot(k, Energy, label="HW $256^2$")

(k, Energy) = np.loadtxt("./HW_N512/energy/Spectrum_1000.txt",dtype="float64",unpack=True)
plt.plot(k, Energy, label="HW $512^2$")

# (k, Energy) = np.loadtxt("./mHW_N1024/energy/Spectrum_0200.txt",dtype="float64",unpack=True)
# plt.plot(k, Energy, label="mHW $1024^2$")

plt.xlabel(r"$k$ $[\rho^{-1}_i]$")
plt.ylabel(r"$\mathcal{F} (\phi)$")

plt.legend(frameon=False)
plt.savefig('spectra.png')
plt.close()
