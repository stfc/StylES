import os
import numpy as np

NDNS = 100
RESTART = False

np.random.seed(0)

for n in range(NDNS):
  r1 = np.random.random()       # seed mixmode x
  r2 = np.random.random()       # seed mixmode z

  cont = n%4

  r3 = 1.0
  r4 = 0.5

  #if (cont==0):
  #  r3 = 1.0 #10**np.random.uniform(-3,2)   # alpha
  #  r4 = 0.5 #10**np.random.uniform(-2,1)   # kappa 
  #if (cont==1):
  #  r3 = 1.0
  #  r4 = 1.0
  #if (cont==2):
  #  r3 = 1.0
  #  r4 = 1.0
  #if (cont==3):
  #  r3 = 1.0
  #  r4 = 1.0

  
  newfolder = "HW_data/HW_" + str(n)

  if (RESTART):
    cmd = "sed -i 's/atani/atani restart/g' " + newfolder + "/run.sh"
    os.system(cmd)

    cmd = "cd " + newfolder + "; bsub < run.sh "
    os.system(cmd)
  else:
    if (not os.path.isfile(newfolder + "/data/BOUT.dmp.0.nc")):
      cmd = ("cp -r HW " + newfolder)
      os.system(cmd)

      cmd = "sed -i s/AAA/" + str(r1) + "/g " + newfolder + "/data/BOUT.inp"
      os.system(cmd)

      cmd = "sed -i s/BBB/" + str(r2) + "/g " + newfolder + "/data/BOUT.inp"
      os.system(cmd)

      cmd = "sed -i s/CCC/" + str(r3) + "/g " + newfolder + "/data/BOUT.inp"
      os.system(cmd)

      cmd = "sed -i s/DDD/" + str(r4) + "/g " + newfolder + "/data/BOUT.inp"
      os.system(cmd)

      cmd = "cd " + newfolder + "; sbatch run.sh "
      os.system(cmd)
      
 

