import argparse

import numpy as np 
import struct
import chainer

import speaker_net as net

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

    encode = []

    for ind in range(0, len(buf), 64 * 4):
        encode.append(struct.unpack('f' * 64, buf[ind:ind+64*4]))

    #   Data to parse
    data = np.asarray(encode, dtype=np.float32)

    #   Place for encoded data
    enc = []

    #   Encode
    speaker = model.predictor(chainer.Variable(data, volatile='on')).data

    for s in speaker:
        arg.output.write(struct.pack('f' * 16, *s))

    arg.output.flush()
    print(time.time() - start)