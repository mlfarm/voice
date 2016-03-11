#   encode:
import data
import os
import numpy as np
import fft_net
import chainer
import struct

model = fft_net.load_latest()

files = os.listdir('fft')

for f in files:
    ffts = []

    print(f)

    fin = open(os.path.join('fft', f), 'rb')
    fout = open('fft-encode/{}.repre'.format(f.split('.')[0]), 'wb')

    buf = fin.read(1024 * 4)

    while len(buf) != 0:
        x = chainer.Variable(np.asarray([struct.unpack('f' * 1024, buf)], dtype=np.float32))
        enc = model.encode(x).data[0]
        fout.write(struct.pack('f' * 64, *enc))
        fout.flush()

        buf = fin.read(1024 * 4)
    fin.close()