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
import codecs

import chainer
from chainer import optimizers
from chainer import serializers

import encode_net as net

#   Load latest model
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

def load_power_data(filepath):
    fin = open(filepath, 'rb')
    buf = fin.read()
    fin.close()

    ffts = []

    for ind in range(0, len(buf), 1024 * 4):
        ffts.append(struct.unpack('f' * 1024, buf[ind:ind+1024*4]))

    return np.asarray(ffts, dtype=np.float32)

def encode(inpath, outpath):
    print("Encodeing: {} -> {}".format(inpath, outpath))
    
    #   Load data
    data = load_power_data(inpath)

    #   Reset model
    model.reset_state()

    #   Place for encoded data
    enc = np.ndarray(shape=(data.shape[0], 64), dtype=np.float32)

    #   Encode
    for i in range(data.shape[0]):
        x = chainer.Variable(np.asarray([data[i]]), volatile='on')
        enc[i] = model.encode2(model.encode1(x)).data[0]

    #   Save encoded data
    np.save(outpath, enc)


class SpeakerDatabase(object):
    def __init__(self, speakerIndex='speaker/speaker_index.txt', rawFileIndex='speaker/file_index.txt'):
        self.index = speakerIndex
        self.fileIndex = rawFileIndex

    def register(self, alias, name):
        fp = codecs.open(self.index, 'r+', 'utf-8')

        for line in fp:
            if line.split()[0] == alias:
                return False

        fp.write("{} {}\n".format(alias, name))
        fp.close()

    def findByAlias(self, alias):
        fp = codecs.open(self.index, 'r', 'utf-8')

        index = 0
        for line in fp:
            if line.split()[0] == alias:
                fp.close()
                return line.split()[1], index
            index += 1

        fp.close()
        return None

    def addFile(self, alias, filepath):
        shutil.move(filepath, os.path.join('speaker/raw', os.path.basename(filepath)))
        fp = codecs.open(self.fileIndex, 'a', 'utf-8')
        fp.write("{} {}\n".format(alias, os.path.basename(filepath)))
        fp.close()

