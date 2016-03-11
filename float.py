#   encode: utf-8

import numpy as np
import os
import wave
import struct

files = os.listdir('wave')

for f in files:
    print("Parsing {}".format(f))

    wf = wave.open(os.path.join('wave', f))
    ff = open(os.path.join('float', f.split('.')[0] + '.float'), 'wb')

    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype='int16') / 32768.0

    wf.close()

    ff.write(struct.pack('f' * len(x), *x))