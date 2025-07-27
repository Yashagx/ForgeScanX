import os
for f in os.listdir("data/results/CoMoFoD/visual"):
    if "_F" in f:
        print(f)