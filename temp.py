import voice
import chainer
import numpy as np
import os
import struct
import threading
import time

files = os.listdir('data/power')

threadCount = 0

def encode(filename):
    global threadCount
    start = time.time()
    data = voice.load_power('data/power/' + filename)
    enco = voice.encode(data)
    fout = open('data/encode/' + filename + '.encode', 'wb')

    enco = enco.ravel()

    fout.write(struct.pack('f' * len(enco), *enco))
    fout.close()
    threadCount -= 1
    print("Process Time: {}".format(time.time() - start))

for filename in files:
    print(filename)
    start = time.time()
    data = voice.load_power('data/power/' + filename)
    enco = voice.encode(data)
    fout = open('data/encode/' + filename + '.encode', 'wb')

    enco = enco.ravel()

    fout.write(struct.pack('f' * len(enco), *enco))
    fout.close()
    print("Process Time: {}".format(time.time() - start))