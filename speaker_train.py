import numpy as np
import chainer
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import speaker_data as data
import speaker_net as net

#   Configuration
n_epoch = 20
batchsize = 100

#   Load model
model, stamp = net.load_latest()

#   Setup Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

#   Load learning Data
for i in range(100):
    print("Loading Data")
    d, x0, x1 = data.load(80000)
    print("Done")

    d_train, d_test = np.split(d, [60000])
    x0_train, x0_test = np.split(x0, [60000])
    x1_train, x1_test = np.split(x1, [60000])

    N = len(d_train)

    #   Evaluation
    def evaluation(d, x0, x1):
        d = chainer.Variable(np.asarray(d, dtype=np.int32), volatile='on')
        x0 = chainer.Variable(np.asarray(x0, dtype=np.float32), volatile='on')
        x1 = chainer.Variable(np.asarray(x1, dtype=np.float32), volatile='on')

        return model(x0, x1, d).data / len(d)

    for epoch in range(n_epoch):
        print("Epoch {}".format(epoch))

        perm = np.random.permutation(N)

        sum_loss = 0

        for i in range(0, N, batchsize):
            d = chainer.Variable(np.asarray(d_train[i:i+batchsize], dtype=np.int32))
            x0 = chainer.Variable(np.asarray(x0_train[i:i+batchsize], dtype=np.float32))
            x1 = chainer.Variable(np.asarray(x1_train[i:i+batchsize], dtype=np.float32))

            optimizer.update(model, x0, x1, d)

            sum_loss += model.loss.data

        print("Train data loss {}".format(sum_loss / N))

        print("Evaluation loss {}".format(evaluation(d_test, x0_test, x1_test)))

    print("Save")
    net.save(model)