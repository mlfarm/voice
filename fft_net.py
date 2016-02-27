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
d_fft_model = 'fft-model'

def load_new():
    return autoencoder.RMLAE(dims=(1024, 256, 64, 256), noise=0.1)

def load_latest():
    params = os.listdir(d_fft_model)
    params.sort()

    model = load_new()

    if len(params) > 0:
        serializers.load_hdf5(os.join(d_fft_model, params[-1]), model)

    return model

def save(model):
    serializers.save_hdf2(os.join(d_fft_model, '{}'.format(int(time.time())), model))