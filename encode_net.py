#   encode: utf8

import os
import time
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers

class Encoder(chainer.Chain):
    def __init__(self, train=True):
        super(Encoder, self).__init__(
            enc1=L.LSTM(1024, 256),
            enc2=L.Linear(256, 64),
            dec=L.Linear(64, 1024)
        )

        self.train = train

    def encode(self, x):
        h = self.enc1(x)
        return F.relu(self.enc2(h))

    def decode(self, y):
        return F.relu(self.dec(y))

    def __call__(self, x, t):
        self.y = self.encode(x)
        self.t = self.decode(self.y)

        return F.mean_squared_error(t, self.t)

    def reset_state(self):
        self.enc1.reset_state()
    
def load_new(train=True):
    return Encoder(train)

def load_latest(train=True):
    model = load_new(train)

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
