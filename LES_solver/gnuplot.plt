set term jpeg
set output "energy.jpeg"

set logscale x
set logscale y

plot for [i=0:1000:100] 'Energy_spectrum_'.i.'.txt' using 1:2 with lines title 't'.i
