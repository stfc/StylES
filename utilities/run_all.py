import  os

# for i in range(0,100):
#   dest = "LES_Solvers_" + str(i)
#   cmd = "cp -r LES_Solvers " + dest
#   #print(cmd)

#   os.system(cmd)
#   cmd = "sed -i 's/AAA/" + str(i) + "/g' LES_Solvers_" + str(i) + "/testcases/HIT_2D/HIT_2D.py"
#   #print(cmd)
#   os.system(cmd)

#   os.chdir(dest)
#   cmd = "bsub < run.sh"
#   os.system(cmd)
#   os.chdir("../")


cont=0
for i in range(0,100):
  source = "./LES_Solvers_" + str(i) + "/fields/"
  dest   = "./LES_Solvers/fields/"

  files = os.listdir(source)
  nfiles = len(files)
  for i,file in enumerate(sorted(files)):
      sfilename = source + file
      tail = file.replace("run0","run" + str(cont))
      dfilename = dest + tail

      cmd = "mv " + sfilename + " " + dfilename
      print(cmd)
      os.system(cmd)

  cont = cont+1
