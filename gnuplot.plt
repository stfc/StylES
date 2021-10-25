#set terminal svg
#set output "./utilities/energy.svg"
#set terminal svg enhanced background rgb 'white'

set logscale x
set logscale y

set yrange [1e-10:1]
set xrange [10:1e4]
#set yrange [1e-10:0.05]
#set xrange [10:700]

set grid

set style circle radius screen 0.003



#plot for [i=0:1000:100] 'Energy_spectrum_'.i.'.txt' using 1:2 with lines title 't'.i


# plot '../results/2D_DNS_ReT60_N8192/Energy_spectrum_9te.txt'    using 1:2 with lines title '9te',    \
#      '../results/2D_DNS_ReT60_N8192/Energy_spectrum_24te.txt'   using 1:2 with lines title '24te',    \
#      '../results/2D_DNS_ReT60_N8192/Energy_spectrum_97te.txt'   using 1:2 with lines title '97te',   \
#      '../results/2D_DNS_ReT60_N8192/Energy_spectrum_134te.txt'  using 1:2 with lines title '134te',   \
#      'testcases/HIT_2D/ld_spectrum_9te.txt'                  using 1:2 with circles title 't = 0.9 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_24te.txt'                 using 1:2 with circles title 't = 0.24 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_97te.txt'                 using 1:2 with circles title 't = 0.97 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_134te.txt'                using 1:2 with circles title 't = 0.134 (L\&D)' 
#      # 'testcases/HIT_2D/ld_spectrum_9te.txt'                  using 1:(1e5*$1**(-3)) with lines lc 'black' title 'k^-3', \
#      # 'testcases/HIT_2D/ld_spectrum_9te.txt'                  using 1:(1e5*$1**(-4)) with lines lc 'black' title 'k^-4'


# plot '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N256/Energy_spectrum_9te.txt'   using 1:2 with lines title 'N256',    \
#      '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N512/Energy_spectrum_9te.txt'   using 1:2 with lines title 'N512',    \
#      '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N1024/Energy_spectrum_9te.txt'  using 1:2 with lines title 'N1024',   \
#      '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N2048/Energy_spectrum_9te.txt'  using 1:2 with lines title 'N2048',   \
#      '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N4096/Energy_spectrum_9te.txt'  using 1:2 with lines title 'N4096',   \
#      '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N8192/Energy_spectrum_9te.txt'  using 1:2 with lines title 'N8192',   \
#      './LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'                            using 1:2 with circles title 't = 9te (L\&D)'


# plot 'DNS_center_values.txt' using 1:2 with lines title 'DNS_org U', \
#      'DNSfromLES_center_values.txt' using 1:2 with lines title 'DNS_fromLES U', \
#      'LES_center_values.txt' using 1:2 with lines title 'LES U'


plot './utilities/Energy_spectrum_it2.txt'   using 1:2 with lines title '16x16',   \
     './utilities/Energy_spectrum_it3.txt'   using 1:2 with lines title '32x32',   \
     './utilities/Energy_spectrum_it4.txt'   using 1:2 with lines title '64x64',   \
     './utilities/Energy_spectrum_it5.txt'   using 1:2 with lines title '128x128', \
     './utilities/Energy_spectrum_it6.txt'   using 1:2 with lines title '256x256', \
     '../../results/decayisoturb_2D/DNS/2D_DNS_ReT60_N256/Energy_spectrum_97te.txt' using 1:2 with circles title 't = 97te'


#------------------------------check latent space
# do for [i=0:99] {
#    filename = sprintf('spectrum_%d.png',i)
#    set output filename
#    plot './energy_spectrum_latents/Energy_spectrum_lat_'.i.'_res_16.txt'   using 1:2 with lines title '16', \
#         './energy_spectrum_latents/Energy_spectrum_lat_'.i.'_res_32.txt'   using 1:2 with lines title '32', \
#         './energy_spectrum_latents/Energy_spectrum_lat_'.i.'_res_64.txt'   using 1:2 with lines title '64', \
#         './energy_spectrum_latents/Energy_spectrum_lat_'.i.'_res_128.txt'  using 1:2 with lines title '128', \
#         './energy_spectrum_latents/Energy_spectrum_lat_'.i.'_res_256.txt'  using 1:2 with lines title '256', \
#         './Energy_spectrum_N256_9te.txt'                                      using 1:2 with circles title '9\t_e   L\&D', \
#         './Energy_spectrum_N256_24te.txt'                                     using 1:2 with circles title '24\t_e  L\&D', \
#         './Energy_spectrum_N256_97te.txt'                                     using 1:2 with circles title '97\t_e  L\&D', \
#         './Energy_spectrum_N256_134te.txt'                                    using 1:2 with circles title '134\t_e L\&D'
# }

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


