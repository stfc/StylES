import os

for i in range(0,100):
  job = 14724196 + i
  cmd = "bkill " + str(job)
  os.system(cmd)  
  cmd = "rm -rf LES_Solvers_" + str(i)
  os.system(cmd)
