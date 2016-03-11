#   encode: utf-8

import numpy as np
import os
import wave
import struct
import subprocess

files = os.listdir('float')

for f in files:
    print("Parsing {}".format(f))

    input = os.path.join('float', f);
    output = os.path.join('fft', f.split('.')[0] + '.fft');

    print("parsing {} -> {}".format(input, output))

    proc = subprocess.Popen('frame -l 1024 -p 256 < {} | window -l 1024 | fftr -l 1024 -P > {}'.format(input, output), stdout=subprocess.PIPE, shell=True)
    proc.wait()
