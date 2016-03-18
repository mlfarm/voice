
import chainer
import chainer.functions as F
import chainer.links as L

class SpeakerNN(chainer.Chain):
    def __init__(self, train=True):
        super(SpeakerNN, self).__init__(
            l1=L.Linear(64, 32),
            l2=L.Linear(32, 32),
            l3=L.Linear(32, 16)
        )

        self.train = train

    def __call__(self, x):
        h1 = F.relu(self.l1(F.dropout(x, train=self.train)))
        h2 = F.relu(self.l2(F.dropout(h1, train=self.train)))
        self.y = F.relu(self.l3(F.dropout(h2, train=self.train)))

        return self.y

def load_new(train=True):
    return SpeakerNN(train)

def load_latest(train=True):
    model = load_new(train)

    