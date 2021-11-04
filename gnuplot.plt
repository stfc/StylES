set terminal png
set output "energy.png"

set logscale x
set logscale y

#set format y "10^{%T}"

set yrange [1e-8:0.2]
set xrange [10:3000]
#set yrange [1e-10:0.05]
#set xrange [10:700]

set grid

set style circle radius screen 0.003



# plot 'LES_Solvers/energy_spectrum_0te.txt'                using 1:2 with lines title '0te',  \
#      'LES_Solvers/energy_spectrum_9te.txt'                using 1:2 with lines title '9te',  \
#      'LES_Solvers/energy_spectrum_24te.txt'               using 1:2 with lines title '24te', \
#      'LES_Solvers/energy_spectrum_97te.txt'               using 1:2 with lines title '97te',  \
#      'LES_Solvers/energy_spectrum_134te.txt'              using 1:2 with lines title '134te', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_0te.txt'   using 1:2 with circles title '0te (L\&D)', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'   using 1:2 with circles title '9te (L\&D)', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_24te.txt'  using 1:2 with circles title '24te (L\&D)', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_97te.txt'  using 1:2 with circles title '97te (L\&D)', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt' using 1:2 with circles title '134te (L\&D)'
# #      'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-3)) with lines lc 'red' title 'k^-3', \
# #      'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-4)) with lines lc 'black' title 'k^-4'

# filename = 'spectrum_134te.png'
# set output filename
# plot '../../results/decayisoturb_2D/DNS/second_order_scheme/N256/energy_spectrum_134te.txt'           using 1:2 with lines title '134te N256',  \
#      '../../results/decayisoturb_2D/DNS/second_order_scheme/N512/energy_spectrum_134te.txt'           using 1:2 with lines title '134te N512',  \
#      '../../results/decayisoturb_2D/DNS/second_order_scheme/N1024/energy_spectrum_134te.txt'          using 1:2 with lines title '134te N1024',  \
#      '../../results/decayisoturb_2D/DNS/second_order_scheme/N2048/energy_spectrum_134te.txt'          using 1:2 with lines title '134te N2048',  \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt'   using 1:2 with circles title '134te (L\&D)'

     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N256/energy_spectrum_24te.txt'           using 1:2 with lines title '24te N256',  \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N512/energy_spectrum_24te.txt'           using 1:2 with lines title '24te N512',  \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N1024/energy_spectrum_24te.txt'          using 1:2 with lines title '24te N1024',  \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N2048/energy_spectrum_24te.txt'          using 1:2 with lines title '24te N2048',  \
     # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_24te.txt'   using 1:2 with circles title '24te (L\&D)'
     
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N256/energy_spectrum_97te.txt'           using 1:2 with lines title '97te N256',  \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N512/energy_spectrum_97te.txt'           using 1:2 with lines title '97te N512',  \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N1024/energy_spectrum_97te.txt'          using 1:2 with lines title '97te N1024',  \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N2048/energy_spectrum_97te.txt'          using 1:2 with lines title '97te N2048',  \
     # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_97te.txt'   using 1:2 with circles title '97te (L\&D)' , \

     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N256/energy_spectrum_134te.txt'          using 1:2 with lines title '134te N256' , \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N512/energy_spectrum_134te.txt'          using 1:2 with lines title '134te N512', \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N1024/energy_spectrum_134te.txt'         using 1:2 with lines title '134te N1024', \
     # '../../results/decayisoturb_2D/DNS/second_order_scheme/N2048/energy_spectrum_134te.txt'         using 1:2 with lines title '134te N2048', \
     # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt'  using 1:2 with circles title '134te (L\&D)'
     # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-3)) with lines lc 'red' title 'k^-3', \
     # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-4)) with lines lc 'black' title 'k^-4'




# #------------------------------check latent space
# filename = 'utilities/spectrum_16.png'
# set output filename
# plot 'utilities/energy_spectrum_lat_0_res_1024.txt'   using 1:($2*400) with lines title '1024', \
#      'utilities/energy_spectrum_lat_0_res_512.txt'   using 1:($2*400) with lines title '512', \
#      'utilities/energy_spectrum_lat_0_res_256.txt'   using 1:($2*400) with lines title '256', \
#      'utilities/energy_spectrum_lat_0_res_128.txt'   using 1:($2*400) with lines title '128', \
#      'utilities/energy_spectrum_lat_0_res_64.txt'   using 1:($2*400) with lines title '64', \
#      'utilities/energy_spectrum_lat_0_res_32.txt'   using 1:($2*400) with lines title '32', \
#      'utilities/energy_spectrum_lat_0_res_16.txt'   using 1:($2*400) with lines title '16', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt' using 1:2 with circles title '134\t_e L\&D'

#      # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_0te.txt' using 1:2 with circles title '0\t_e L\&D'
#      # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt' using 1:2 with circles title '9\t_e L\&D'
#      # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_24te.txt' using 1:2 with circles title '24\t_e L\&D'
#      # 'LES_Solvers/testcases/HIT_2D/ld_spectrum_97te.txt' using 1:2 with circles title '97\t_e L\&D'

do for [i=0:9] {
   filename = sprintf('utilities/spectrum_%d.png',i)
   set output filename
   plot 'utilities/latents/energy_spectrum_lat_'.i.'_res_64.txt'   using 1:($2*400) with lines title '64', \
        'utilities/latents/energy_spectrum_lat_'.i.'_res_128.txt'  using 1:($2*400) with lines title '128', \
        'utilities/latents/energy_spectrum_lat_'.i.'_res_256.txt'  using 1:($2*400) with lines title '256', \
        'utilities/latents/energy_spectrum_lat_'.i.'_res_512.txt'  using 1:($2*400) with lines title '512', \
        'utilities/latents/energy_spectrum_lat_'.i.'_res_1024.txt'  using 1:($2*400) with lines title '1024', \
        'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt' using 1:2 with circles title '134\t_e L\&D'
}

#------------------------------check styles
#do for [i=0:13] {
#   filename = sprintf('spectrum_%d.png',i)
#   set output filename
#   plot './energy_spectrum_styles/Energy_spectrum_styles_'.i.'_level_0.txt'   using 1:2 with lines title '0', \
#        './energy_spectrum_styles/Energy_spectrum_styles_'.i.'_level_1.txt'   using 1:2 with lines title '1', \
#        './energy_spectrum_styles/Energy_spectrum_styles_'.i.'_level_2.txt'   using 1:2 with lines title '2', \
#        './energy_spectrum_styles/Energy_spectrum_styles_'.i.'_level_3.txt'   using 1:2 with lines title '3', \
#        './energy_spectrum_styles/Energy_spectrum_styles_'.i.'_level_4.txt'   using 1:2 with lines title '4', \
#       './Energy_spectrum_N256_9te.txt'                                      using 1:2 with circles title '9\t_e   L\&D', \
#       './Energy_spectrum_N256_24te.txt'                                     using 1:2 with circles title '24\t_e  L\&D', \
#       './Energy_spectrum_N256_97te.txt'                                     using 1:2 with circles title '97\t_e  L\&D', \
#       './Energy_spectrum_N256_134te.txt'                                    using 1:2 with circles title '134\t_e L\&D'
#}
#


