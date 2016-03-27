import argparse

import numpy as np 
import struct
import chainer

import encode_net as net

import time

model, timestamp = net.load_latest()


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=argparse.FileType('rb'))
    parser.add_argument('output', type=argparse.FileType('wb'))

    arg = parser.parse_args()

    buf = arg.input.read()
    arg.input.close()

    ffts = []

    for ind in range(0, len(buf), 1024 * 4):
        ffts.append(struct.unpack('f' * 1024, buf[ind:ind+1024*4]))

    #   Data to parse
    data = np.asarray(ffts, dtype=np.float32)

    #   Place for encoded data
    enc = []

    #   Encode
    for i in range(data.shape[0]):
        x = chainer.Variable(np.asarray([data[i]]), volatile='on')
        enc.extend(model.encode2(model.encode1(x)).data[0])

    arg.output.write(struct.pack('f' * len(enc), *enc))
    print(time.time() - start)