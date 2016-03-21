#   encode: utf-8

from datetime import datetime
from pytz import timezone  
import os
import shutil
import subprocess
import time
import wave
import numpy as np
import struct

import chainer
from chainer import optimizers
from chainer import serializers

import encode_net as net

model, timestamp = net.load_latest()


def convert2wav(inpath, outpath):
    print("Converting to wav: {} -> {}".format(inpath, outpath))
    result = subprocess.call("ffmpeg -y -i {} -ac 1 -ar 44100 {}".format(inpath, outpath), 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result == 0:
        return True
    else:
        return False

def convert2float(inpath, outpath):
    print("Converting to float: {} -> {}".format(inpath, outpath))
    #   Convert to float
    wf = wave.open(inpath)

    #   read data
    x = wf.readframes(wf.getnframes())

    #   close file
    wf.close()

    #   convert to [-1, 1]
    x = np.frombuffer(x, dtype='int16') / 32768.0

    #   write
    ff = open(outpath, 'wb')
    ff.write(struct.pack('f' * len(x), *x))
    ff.close()

    return os.path.isfile(outpath)

def convert2power(inpath, outpath):
    print("Converting to power: {} -> {}".format(inpath, outpath))
    
    result = subprocess.call('frame -l 1024 -p 256 < {} | window -l 1024 | fftr -l 1024 -P > {}'.format(inpath, outpath), 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result == 0:
        return True
    else:
        return False

def load_data(filename):
    fin = open(filename, 'rb')
    buf = fin.read()
    fin.close()

    ffts = []

    for ind in range(0, len(buf), 1024 * 4):
        ffts.append(struct.unpack('f' * 1024, buf[ind:ind+1024*4]))

    return np.asarray(ffts, dtype=np.float32)

def encode(inpath, outpath):
    #   Load data
    data = load_data(inpath)

    #   Reset model
    model.reset_state()

    #   Place for encoded data
    enc = np.ndarray(shape=(data.shape[0], 64), dtype=np.float32)

    #   Encode
    for i in range(d.shape[0]):
        x = chainer.Variable(np.asarray([d[i]]), volatile='on')
        enc[i] = model.encode2(model.encode1(x)).data[0]

    #   Save encoded data
    np.save(outpath, enc)

def load(audioURL):
    #   Get Basename
    basename = '.'.join(os.path.basename(audioURL).split('.')[:-1])

    #   Download
    res = subprocess.call('wget {} -O {}'.format(audioURL, os.path.join('tmp', basename + '.mp3')), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if res != 0:
        return False

    #   Convert .mp3 to .wav
    convert2wav(os.path.join('tmp', basename + '.mp3'), os.path.join('tmp', basename + '.wav'))

    #   .wav to .float
    convert2float(os.path.join('tmp', basename + '.wav'), os.path.join('tmp', basename + '.float'))

    #   .float to .power
    convert2power(os.path.join('tmp', basename + '.float'), os.path.join('tmp', basename + '.power'))

    #   Encode
    encode(os.path.join('tmp', basename + '.power'), os.path.join('encoded', basename))

    #   Erase anything in tmp file
    os.system("rm tmp/*")

    return True

def load_talent(talent):
    #   フリートーク
    time.sleep(1)
    load('http://www.sigma7.co.jp/profile/mp3/{}_f.mp3'.format(talent))

    #   ナレーション
    index = 1
    while True:
        time.sleep(1.0)
        if load('http://www.sigma7.co.jp/profile/mp3/{}_{0:0>2}.mp3'.format(talent, index)):
            index += 1
        else:
            break
