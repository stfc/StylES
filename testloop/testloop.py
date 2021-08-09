import os

listcase = ["2D_HIT", 
            "divergence"]

for testcase in listcase:
    print("\n\n ======================== Running test " + testcase + " ========================\n")

    cmd1 = "sed -i 's/CASE/" + testcase + "/g' parameters.py"
    cmd2 = "sed -i 's/CASE/" + testcase + "/g' testloop/compare.py"
    os.system(cmd1)
    os.system(cmd2)

    os.system("python main.py")
    os.system("cd testloop && python compare.py")

    cmd1 = "sed -i 's/" + testcase + "/CASE/g' parameters.py"
    cmd2 = "sed -i 's/" + testcase + "/CASE/g' testloop/compare.py"
    os.system(cmd1)
    os.system(cmd2)
