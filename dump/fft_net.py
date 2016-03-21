# encode: utf-8

"""
    学習したパラメータを管理する
"""
import os
import time

import chainer
from chainer import serializers

import autoencoder

# Directory
d_fft_model = 'data/fft-model'

def load_new():
    return autoencoder.RMLAE(dims=(1024, 256, 64, 256), noise=0.1)

def load_latest():
    params = os.listdir(d_fft_model)
    params.sort()

    model = load_new()

    if len(params) > 0:
        serializers.load_hdf5(os.path.join(d_fft_model, params[-1]), model)
        return model, params[-1]
    else:
        return model, None

def save(model):
    path = os.path.join(d_fft_model, '{}'.format(int(time.time())))
    serializers.save_hdf5(path, model)