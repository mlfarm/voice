# encode: utf-8

import numpy as np 
import os
import struct

files = os.listdir('data/power')
perm  = np.random.permutation(len(files))

index = 0

def load():
    global index

    #   Storage
    ffts = []

    #   Open File
    fin = open(os.path.join('data/power', files[perm[index]]), 'rb')
    index += 1

    #   Read contents
    buf = fin.read()

    #   Close file
    fin.close()

    for ind in range(0, len(buf), 1024 * 4):
        ffts.append(struct.unpack('f' * 1024, buf[ind:ind+1024*4]))

    return np.asarray(ffts, dtype=np.float32)
