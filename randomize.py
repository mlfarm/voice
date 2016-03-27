import os
import struct
import numpy as np
import threading


fin = open('speaker/labeled.bin', 'rb')
fout = open('speaker/random.bin', 'wb')

size = os.path.getsize('speaker/labeled.bin')

whole_len = int(size / 258)

perm = np.random.permutation(whole_len)

for index in range(whole_len):
    fin.seek(perm[index] * 258)
    fout.write(fin.read(258))

    if index % 1000 == 0:
        fout.flush()
        print("{} / {} = {:.2}".format(index, whole_len, index / whole_len))