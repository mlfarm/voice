# encode: utf-8

import numpy as np 
import os
import struct

files = os.listdir('data/power')
perm  = np.random.permutation(len(files))

index = 0

def load(n):
    global index

    #   Storage
    ffts = []

    for i in range(n):
        #   Open File
        fin = open(os.path.join('data/power', files[perm[index]]), 'rb')
        index += 1

        #   Read contents
        buf = fin.read()

        #   Close file
        fin.close()

        x = np.frombuffer(buf, dtype=np.float32)

        ffts.extend(x)
        index += 1
    
    d = np.log(np.asarray(ffts, dtype=np.float32).reshape((int(len(ffts) / 1024), 1024))+1)
    return d
