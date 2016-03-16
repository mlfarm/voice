#   encode: utf8

import os
import time
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers

class RAE(chainer.Chain):
    def __init__(self, dims, train=True, noise=0.0, activation=None):
        super(RAE, self).__init__(
            enc=L.LSTM(dims[0], dims[1]),
            dec=L.Linear(dims[1], dims[0])
        )

        self.train = train
        self.noise = noise
        self.activation = activation

    def __call__(self, x, t):
        self.y = self.enc(F.dropout(x, train=self.train))
        self.t = self.dec(F.dropout(self.y, train=self.train))

        self.loss = F.mean_squared_error(t, self.t)

        return self.loss

def load_new():
    return RAE(dims=(1024, 256), noise=0.01)

def load_latest():
    model = load_new()

    params = os.listdir('encode-model')
    params.sort()

    if len(params) > 0:
        serializers.load_hdf5(os.path.join('encode-model', params[-1]), model)
        return model, params[-1]
    else:
        return model, None

def save(model):
    path = os.path.join('encode-model', '{}'.format(int(time.time())))
    serializers.save_hdf5(path, model)
