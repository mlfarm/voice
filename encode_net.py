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
            enc2=L.LSTM(256, 64),
            dec2=L.Linear(64, 256),
            dec1=L.Linear(256, 1024)
        )

        self.train = train

    def layer1(self, x, t):
        self.y1 = self.enc1(F.dropout(x, train=self.train))
        self.t1 = self.dec1(F.dropout(self.y1, train=self.train))

        self.loss1 = F.mean_squared_error(t, self.t1)

        return self.loss1

    def layer2(self, x, t):
        self.y2 = self.enc2(F.dropout(x, train=self.train))
        self.t2 = self.dec2(F.dropout(self.y2, train=self.train))

        self.loss2 = F.mean_squared_error(t, self.t2)

        return self.loss2

    def encode1(self, x):
        return self.enc1(x)

    def encode2(self, x):
        return self.enc2(x)

def load_new():
    return Encoder()

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
