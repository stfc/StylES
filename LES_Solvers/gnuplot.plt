#set terminal svg
#set output "energy.svg"

#set logscale x
#set logscale y

#set yrange [0.64:0.68]
set xrange [0.01:0.0102]

#plot for [i=0:1000:100] 'Energy_spectrum_'.i.'.txt' using 1:2 with lines title 't'.i

set style circle radius screen 0.003

     # 'testcases/HIT_2D/ld_spectrum.txt'          using 1:2 with circles title 't = 0     (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_9te.txt'      using 1:2 with circles title 't = 0.9   (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_24te.txt'     using 1:2 with circles title 't = 0.24  (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_97te.txt'     using 1:2 with circles title 't = 0.97  (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_134te.txt'    using 1:2 with circles title 't = 0.134 (L&D)'

# plot 'results/2D_DNS_ReT60_N4096/Energy_spectrum_9te.txt'    using 1:2 with lines title 'N256',    \
#      'results/2D_DNS_ReT60_N4096/Energy_spectrum_24te.txt'   using 1:2 with lines title 'N512',    \
#      'results/2D_DNS_ReT60_N4096/Energy_spectrum_97te.txt'   using 1:2 with lines title 'N1024',   \
#      'results/2D_DNS_ReT60_N4096/Energy_spectrum_134te.txt'  using 1:2 with lines title 'N2048',   \
#      'Energy_spectrum_555.txt'    using 1:2 with lines title 'sta 256',    \
#      'Energy_spectrum_1289.txt'   using 1:2 with lines title 'sta N512',    \
#      'Energy_spectrum_3492.txt'   using 1:2 with lines title 'sta N1024',   \
#      'Energy_spectrum_4325.txt'  using 1:2 with lines title 'sta N2048',   \
#      'testcases/HIT_2D/ld_spectrum_9te.txt'                  using 1:2 with circles title 't = 0.9 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_24te.txt'                 using 1:2 with circles title 't = 0.24 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_97te.txt'                 using 1:2 with circles title 't = 0.97 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_134te.txt'                 using 1:2 with circles title 't = 0.134 (L\&D)' , \
#      'testcases/HIT_2D/ld_spectrum_9te.txt'                  using 1:(1e5*$1**(-3)) with lines lc 'black' title 'k^-3', \
#      'testcases/HIT_2D/ld_spectrum_9te.txt'                  using 1:(1e5*$1**(-4)) with lines lc 'black' title 'k^-4'

plot 'DNS_center_values.txt' using 1:2 with lines title 'DNS_org U', \
     'DNSfromLES_center_values.txt' using 1:2 with lines title 'DNS_fromLES U', \
     'LES_center_values.txt' using 1:2 with lines title 'LES U'
