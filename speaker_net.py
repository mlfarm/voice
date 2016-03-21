import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import os
import numpy as np
import time

class SpeakerNN(chainer.Chain):
    def __init__(self, train):
        super(SpeakerNN, self).__init__(
            l1=L.Linear(64, 32),
            l2=L.Linear(32, 32),
            l3=L.Linear(32, 16)
        )

        self.train = train

    def __call__(self, x, d):
        x.data = x.data / np.linalg.norm(x.data)
        h1 = F.relu(self.l1(F.dropout(x, self.train)))
        h2 = F.relu(self.l2(F.dropout(h1, self.train)))
        h3 = F.relu(self.l3(F.dropout(h2, self.train)))

        return h3;

class Discriminator(chainer.Chain):
    def __init__(self, predictor):
        super(Discriminator, self).__init__(
            predictor=predictor
        )

    def __call__(self, x0, x1, d):
        y0 = self.predictor(x0)
        y1 = self.predictor(x1)
        self.loss = F.contrastive(y0, y1, d)
        return self.loss


def load_new(train=True):
    return Discriminator(SpeakerNN(train))

def load_latest(train=True):
    model = load_new(train)

    params = os.listdir('speaker-model')
    params.sort()

    if len(params) > 0:
        serializers.load_hdf5(os.path.join('speaker-model', params[-1]), model)
        return model, params[-1]
    else:
        return model, None

def save(model):
    path = os.path.join('speaker-model', '{}'.format(int(time.time())))
    serializers.save_hdf5(path, model)