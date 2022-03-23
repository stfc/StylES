#system ("rm energy.png")

# set terminal png size 1024,768
# set output "energy.png"

#set logscale x
#set logscale y

#set format y "10^{%T}"

#set xrange [10:3000]
#set yrange [1e-12:0.2]

#set yrange [1e-4:0.1]
set xrange [0.06:0.07]

set xrange [0.069:0.075]
set yrange [-2:0.5]

set grid

set style circle radius screen 0.003

# plot './LES_Solvers/results/LES_N256_rs8/DNS_fromGAN_center_values.txt' using 1:2 with lines title   'DNS ', \
#      './LES_Solvers/results/LES_N256_rs8/LES_center_values.txt'         using 1:2 with lines linecolor 5 title   'LES ', \
#      './LES_Solvers/results/LES_N256_rs8/LES_fromGAN_center_values.txt' using 1:2 every 20 w circles t 'LES from DNS'

plot './LES_Solvers/paper_results/tollm5/DNS_N256_1000it/DNS_center_values.txt'        using 1:2 with lines title 'DNS', \
     './LES_Solvers/paper_results/tollm5/DNS_N256_rs8/LES_fromGAN_center_values.txt'   using 1:2 with lines title 'LES from GAN', \
     './LES_Solvers/paper_results/tollm5/A1_N256_rs8_DNStollM4/LES_center_values.txt'  using 1:2 with lines title 'LES tollM4', \
     './LES_Solvers/LES_center_values.txt'  using 1:2 with lines title 'LES tollM5'

# plot './LES_Solvers/results/curves/n128_rs32/LES_fromGAN_center_values.txt'  using 1:2 with lines title   'LES from DNS rs32', \
#      './LES_Solvers/results/curves/n128_rs32/LES_center_values.txt'          using 1:2 with circles title 'LES 128 rs32', \
#      './LES_Solvers/results/curves/n64_rs32/LES_center_values.txt'           using 1:2 with circles title 'LES 64 rs32', \
#      './LES_Solvers/results/curves/n32_rs32/LES_center_values.txt'           using 1:2 with circles title 'LES 32 rs32', \
#      './LES_Solvers/results/curves/n16_rs32/LES_center_values.txt'           using 1:2 with circles title 'LES 16 rs32', \
#      './LES_Solvers/results/curves/n8_rs32/LES_center_values.txt'           using 1:2 with circles title 'LES 8 rs32'


# plot './LES_Solvers/energy/energy_spectrum_DNS_fromGAN_it0.txt' using 1:2 with lines title 'DNS 256', \
#      './LES_Solvers/energy/energy_spectrum_LES_it0.txt'         using 1:2 with lines title 'LES 128', \
#      './LES_Solvers/energy/energy_spectrum_nonlinear_.txt'      using 1:2 with lines title 'UU', \
#      './LES_Solvers/energy/energy_spectrum_filtered_nonlinear_.txt'      using 1:2 with lines title 'fUU'

# plot 'utilities/energy/energy_org_spectrum_lat_0_res_256.txt' using 1:2 with lines title 'DNS 256', \
#      'utilities/energy/energy_org_spectrum_lat_0_res_32.txt' using 1:2 with lines title 'DNS 32', \
#      'utilities/energy/energy_spectrum_lat_0_res_256.txt'     using 1:2 with circles title 'StyleGAN 256', \
#      'utilities/energy/energy_spectrum_lat_0_res_32.txt'     using 1:2 with circles title 'StyleGAN 32', \
#      'utilities/energy/energy_spectrum_fil_lat_0_res_32.txt' using 1:2 with circles title 'filter 32'


# plot 'utilities/energy/energy_org_spectrum_lat_0_res_256.txt' using 1:2 with lines title 'DNS 256', \
#      'utilities/energy/energy_org_spectrum_lat_0_res_128.txt' using 1:2 with lines title 'DNS 128', \
#      'utilities/energy/energy_org_spectrum_lat_0_res_64.txt'  using 1:2 with lines title 'DNS 64', \
#      'utilities/energy/energy_org_spectrum_lat_0_res_32.txt'  using 1:2 with lines title 'DNS 32', \
#      'utilities/energy/energy_org_spectrum_lat_0_res_16.txt'  using 1:2 with lines title 'DNS 16', \
#      'utilities/energy/energy_spectrum_lat_0_res_256.txt'     using 1:2 with lines title 'StyleGAN 256', \
#      'utilities/energy/energy_spectrum_lat_0_res_128.txt'     using 1:2 with lines title 'StyleGAN 128', \
#      'utilities/energy/energy_spectrum_lat_0_res_64.txt'      using 1:2 with lines title 'StyleGAN 64', \
#      'utilities/energy/energy_spectrum_lat_0_res_32.txt'      using 1:2 with lines title 'StyleGAN 32', \
#      'utilities/energy/energy_spectrum_lat_0_res_16.txt'      using 1:2 with lines title 'StyleGAN 16'


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

# do for [i=0:9] {
#    filename = sprintf('utilities/spectrum_%d.png',i)
#    set output filename
#    plot 'utilities/latents/energy_spectrum_lat_'.i.'_res_64.txt'   using 1:($2*400) with lines title '64', \
#         'utilities/latents/energy_spectrum_lat_'.i.'_res_128.txt'  using 1:($2*400) with lines title '128', \
#         'utilities/latents/energy_spectrum_lat_'.i.'_res_256.txt'  using 1:($2*400) with lines title '256', \
#         'utilities/latents/energy_spectrum_lat_'.i.'_res_512.txt'  using 1:($2*400) with lines title '512', \
#         'utilities/latents/energy_spectrum_lat_'.i.'_res_1024.txt'  using 1:($2*400) with lines title '1024', \
#         'LES_Solvers/testcases/HIT_2D/ld_spectrum_134te.txt' using 1:2 with circles title '134\t_e L\&D'
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


