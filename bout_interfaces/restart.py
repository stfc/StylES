import xbout
df = xbout.open_boutdataset("./data/BOUT.dmp.*.nc")
df.bout.to_restart(tind=1)
