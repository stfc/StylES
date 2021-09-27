set terminal svg
set output "energy.svg"

#set logscale x
#set logscale y

set yrange [1e-14:0.05]
set xrange [:600]

#plot for [i=0:1000:100] 'Energy_spectrum_'.i.'.txt' using 1:2 with lines title 't'.i

set style circle radius screen 0.003

     # 'testcases/HIT_2D/ld_spectrum.txt'          using 1:2 with circles title 't = 0     (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_9te.txt'      using 1:2 with circles title 't = 0.9   (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_24te.txt'     using 1:2 with circles title 't = 0.24  (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_97te.txt'     using 1:2 with circles title 't = 0.97  (L&D)', \
     # 'testcases/HIT_2D/ld_spectrum_134te.txt'    using 1:2 with circles title 't = 0.134 (L&D)'

plot 'results/2D_DNS_ReT60_N256/Energy_spectrum_134te.txt'   using 1:2 with lines title 'N256',    \
     'results/2D_DNS_ReT60_N512/Energy_spectrum_134te.txt'   using 1:2 with lines title 'N512',    \
     'results/2D_DNS_ReT60_N1024/Energy_spectrum_134te.txt'  using 1:2 with lines title 'N1024',   \
     'results/2D_DNS_ReT60_N2048/Energy_spectrum_134te.txt' using 1:2 with lines title 'N2048',   \
     'results/2D_DNS_ReT60_N4096/Energy_spectrum_134te.txt' using 1:2 with lines title 'N4096',   \
     'testcases/HIT_2D/ld_spectrum_134te.txt'                using 1:2 with circles title 't = 0.134 (L\&D)'



