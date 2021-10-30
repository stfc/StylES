#set terminal svg
#set output "energy.svg"
#set terminal svg enhanced background rgb 'white'

unset logscale x
unset logscale y

set yrange [1e-3:0.08]
set xrange [10:500]
#set yrange [1e-10:0.05]
#set xrange [10:700]

set grid

set style circle radius screen 0.003



#plot for [i=0:1000:100] 'Energy_spectrum_'.i.'.txt' using 1:2 with lines title 't'.i


plot 'LES_Solvers/N256/energy_spectrum_0te.txt'            using 1:2 with lines title '0te N256',   \
     'LES_Solvers/N256/energy_spectrum_9te.txt'            using 1:2 with lines title '9te N256',  \
     'LES_Solvers/N256/energy_spectrum_24te.txt'           using 1:2 with lines title '24te N256',  \
     'LES_Solvers/N256/energy_spectrum_97te.txt'           using 1:2 with lines title '97te N256',  \
     'LES_Solvers/N256/energy_spectrum_134te.txt'          using 1:2 with lines title '134te N256', \
     'LES_Solvers/N512/energy_spectrum_0te.txt'            using 1:2 with lines title '0te N512',   \
     'LES_Solvers/N512/energy_spectrum_9te.txt'            using 1:2 with lines title '9te N512',   \
     'LES_Solvers/N512/energy_spectrum_24te.txt'           using 1:2 with lines title '24te N512',  \
     'LES_Solvers/N512/energy_spectrum_97te.txt'           using 1:2 with lines title '97te N512',  \
     'LES_Solvers/N512/energy_spectrum_134te.txt'          using 1:2 with lines title '134te N512', \
     'LES_Solvers/N1024/energy_spectrum_0te.txt'           using 1:2 with lines title '0te N1024',   \
     'LES_Solvers/N1024/energy_spectrum_9te.txt'           using 1:2 with lines title '9te N1024',   \
     'LES_Solvers/N1024/energy_spectrum_24te.txt'          using 1:2 with lines title '24te N1024',  \
     'LES_Solvers/N1024/energy_spectrum_97te.txt'          using 1:2 with lines title '97te N1024',  \
     'LES_Solvers/N1024/energy_spectrum_134te.txt'         using 1:2 with lines title '134te N1024', \
     'LES_Solvers/N2048/energy_spectrum_0te.txt'           using 1:2 with lines title '0te N2048',   \
     'LES_Solvers/N2048/energy_spectrum_9te.txt'           using 1:2 with lines title '9te N2048',   \
     'LES_Solvers/N2048/energy_spectrum_24te.txt'          using 1:2 with lines title '24te N2048',  \
     'LES_Solvers/N2048/energy_spectrum_97te.txt'          using 1:2 with lines title '97te N2048',  \
     'LES_Solvers/N2048/energy_spectrum_134te.txt'         using 1:2 with lines title '134te N2048', \
     'LES_Solvers/testcases/HIT_2D/ld_spectrum_0te.txt'    using 1:2 with circles title '0te (L\&D)' , \
     'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:2 with circles title '9te (L\&D)' , \
     'LES_Solvers/testcases/HIT_2D/ld_spectrum_24te.txt'   using 1:2 with circles title '24te (L\&D)' , \
     'LES_Solvers/testcases/HIT_2D/ld_spectrum_97te.txt'   using 1:2 with circles title '97te (L\&D)' , \
     'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt'  using 1:2 with circles title '134te (L\&D)'
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-3)) with lines lc 'red' title 'k^-3', \
#      'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-4)) with lines lc 'black' title 'k^-4'


# plot '../../data/N1024/Energy_spectrum_9te.txt'            using 1:2 with lines title '9te N1024', \
#      '../../data/N1024/Energy_spectrum_24te.txt'           using 1:2 with lines title '24te N1024',    \
#      '../../data/N1024/Energy_spectrum_97te.txt'           using 1:2 with lines title '97te N1024',   \
#      '../../data/N1024/Energy_spectrum_134te.txt'          using 1:2 with lines title '134te N1024',   \
#      'utilities/Energy_spectrum_lat_0_res_16.txt'          using 1:2 with lines title '16 ' , \
#      'utilities/Energy_spectrum_lat_0_res_32.txt'          using 1:2 with lines title '32' , \
#      'utilities/Energy_spectrum_lat_0_res_64.txt'          using 1:2 with lines title '64' , \
#      'utilities/Energy_spectrum_lat_0_res_128.txt'         using 1:2 with lines title '128' , \
#      'utilities/Energy_spectrum_lat_0_res_256.txt'         using 1:2 with lines title '256' , \
#      'utilities/Energy_spectrum_lat_0_res_512.txt'         using 1:2 with lines title '512' , \
#      'utilities/Energy_spectrum_lat_0_res_1024.txt'        using 1:2 with lines title '1024' 
#     #  'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:2 with circles title '9te (L\&D)' , \
#     #  'LES_Solvers/testcases/HIT_2D/ld_spectrum_24te.txt'   using 1:2 with circles title '24te (L\&D)' , \
#     #  'LES_Solvers/testcases/HIT_2D/ld_spectrum_97te.txt'   using 1:2 with circles title '97te (L\&D)' , \
#     #  'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt'  using 1:2 with circles title '134te (L\&D)', \
#     #  'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-3)) with lines lc 'red' title 'k^-3', \
#     #  'LES_Solvers/testcases/HIT_2D/ld_spectrum_9te.txt'    using 1:(1e5*$1**(-4)) with lines lc 'black' title 'k^-4'


# #------------------------------check latent space
# do for [i=0:0] {
#    filename = sprintf('utilities/spectrum_%d.svg',i)
#    set output filename
#    plot 'utilities/energy_spectrum_lat_'.i.'_res_64.txt'   using 1:($2*200) with lines title '64', \
#         'utilities/energy_spectrum_lat_'.i.'_res_128.txt'  using 1:($2*200) with lines title '128', \
#         'utilities/energy_spectrum_lat_'.i.'_res_256.txt'  using 1:($2*200) with lines title '256', \
#         'utilities/energy_spectrum_lat_'.i.'_res_512.txt'  using 1:($2*200) with lines title '512', \
#         'utilities/energy_spectrum_lat_'.i.'_res_1024.txt'  using 1:($2*200) with lines title '1024', \
#         'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt' using 1:2 with circles title '134\t_e L\&D'
# }

#------------------------------check styles
#do for [i=0:13] {
#   filename = sprintf('spectrum_%d.svg',i)
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


