import os

# provide lists for names and decay rate values for each CASE
list_case_name = ["HIT_2D", 
                  "divergence"]

list_case_dr = [1.0e0,
                1.0e0]

cmd1 = "cd testloop && rm *.png"
os.system(cmd1)

for name,dr in zip(list_case_name, list_case_dr):
    print("\n\n ======================== Running test " + name + " ========================\n")

    cmd1 = "sed -i 's/CASE_NAME/"       + name    + "/g' parameters.py"
    cmd2 = "sed -i 's/CASE_DECAY_RATE/" + str(dr) + "/g' parameters.py"
    cmd3 = "sed -i 's/CASE_NAME/"       + name    + "/g' testloop/compare.py"

    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)

    os.system("python main.py")
    os.system("cd testloop && python compare.py")

    cmd1 = "sed -i 's/"                  + name    + "/CASE_NAME/g' parameters.py"
    cmd2 = "sed -i 's/DECAY_RATE     = " + str(dr) + "/DECAY_RATE     = CASE_DECAY_RATE/g' parameters.py"
    cmd3 = "sed -i 's/"                  + name    + "/CASE_NAME/g' testloop/compare.py"

    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
