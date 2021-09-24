#set terminal svg
#set output "energy.svg"

#set logscale x
#set logscale y

#set yrange [1e-14:10000]
set xrange [:600]

#plot for [i=0:1000:100] 'Energy_spectrum_'.i.'.txt' using 1:2 with lines title 't'.i
# plot 'Energy_spectrum_0.txt'    using 1:2 with lines title 't = 0', \
#      'Energy_spectrum_342.txt'  using 1:2 with lines title 't = 0.0342', \
#      'Energy_spectrum_912.txt'  using 1:2 with lines title 't = 0.0912', \
#      'Energy_spectrum_3686.txt' using 1:2 with lines title 't = 0.3686', \
#      'Energy_spectrum_5093.txt' using 1:2 with lines title 't = 0.5093', \
#      'ld_spectrum_9te.txt'      using 1:2 pt 0.5     title 't = 0.9 (L&D)', \
#      'ld_spectrum_24te.txt'     using 1:2 pt 0.5     title 't = 0.24 (L&D)', \
#      'ld_spectrum_97te.txt'     using 1:2 pt 0.5     title 't = 0.97 (L&D)', \
#      'ld_spectrum_134te.txt'    using 1:2 pt 0.5     title 't = 0.134 (L&D)'

set style circle radius screen 0.003

plot 'Energy_spectrum_0.txt'    using 1:2 with lines title 't = 0', \
     'Energy_spectrum_596.txt'  using 1:2 with lines title 't = 0.0342', \
     'Energy_spectrum_1209.txt' using 1:2 with lines title 't = 0.0912', \
     'Energy_spectrum_3983.txt' using 1:2 with lines title 't = 0.0912', \
     'ld_spectrum_9te.txt'      using 1:2 with circles title 't = 0.9 (L&D)', \
     'ld_spectrum_24te.txt'     using 1:2 with circles     title 't = 0.24 (L&D)', \
     'ld_spectrum_97te.txt'     using 1:2 with circles     title 't = 0.97 (L&D)', \
     'ld_spectrum_134te.txt'    using 1:2 with circles     title 't = 0.134 (L&D)'
