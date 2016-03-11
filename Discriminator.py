#   encode: utf-8
import chainer
import chainer.functions as F

class DiscFunc(chainer.Function):
    def forward(self, x):
        dist = x[0]
        d = x[1]

        self.y = np.log(1 + np.exp(dist * d)).astype(np.float32)

        return tuple([self.y])

    def backward(self, input, gy):
        ed = np.exp(input[0] * input[1])

        if input[1] > 0:
            return gy[0] * self.y * ed / (ed + 1), None
        else:
            return - gy[0] * self.y / (ed + 1), None

def discFunc(x, d):
    return DiscFunc()(x, d)

class Discriminator(chainer.Chain):
    def __init__(self, predictor, train):
        super(Discriminator, self).__init__(predictor=predictor)

        self.train = train

    def __call__(self, x1, x2, d):
        self.y1 = self.predictor(x1)
        self.y2 = self.predictor(x2)

        self.norm = F.mean_squared_error(self.y1, self.y2)

        return discFunc(self.norm, d)

    def encode(self, x):
        return self.predictor(x)
