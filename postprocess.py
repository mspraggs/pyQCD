import numpy as np
from os import listdir,system
from os.path import isfile, join
import sys

files = [f for f in listdir("results") if isfile(join("results",f)) and f[-4:] == ".npy"]

files.sort()

if len(files) == 0:
    print("No data available.")
    sys.exit()

print("Available data:")
for i in xrange(len(files)):
    print("(%d) %s" % (i,files[i]))

file_num = input("File: ")
filename = "results/%s" % files[file_num]

Ws = np.load(filename)

Ncf,Nr,Nt = np.shape(Ws)

aVs = np.log(Ws/np.roll(Ws,-1,axis=2))
