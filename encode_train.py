#   encode: utf-8

import numpy as np
import chainer
from chainer import optimizers
from chainer import serializers

import encode_data as data
import encode_net as net

#   Load model
model, timestamp = net.load_latest()

#   Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

#   Evaluation function
def evaluate(dataset):
    evaluator = model.copy()
    evaluator.reset_state()

    sum_loss = 0

    for i in range(dataset.shape[0] - 1):
        x = chainer.Variable(xp.asarray(dataset[i:i+1]), volatile='on')
        t = chainer.Variable(xp.asarray(dataset[i+1:i+2]), volatile='on')

        loss = evaluator(x, t)

        sum_loss += loss.data

    return sum_loss / dataset.shape[0]

#   Train configuration
batchsize = 20
bprop_len = 30
n_epoch = 2
n_refresh = 105

#   Train
for refresh in range(n_refresh):
    print("Loading Data")
    d = data.load()
    print("Done")

    # 学習データの大きさ
    whole_len = d.shape[0]

    # 並列学習の切れ目
    jump = whole_len // batchsize

    # lossの合計
    sum_loss = 0

    # ログ用のloss
    log_loss = 0

    # epoch
    epoch = 0

    # 開始
    print("Iterate: {}".format(jump * n_epoch))
    for i in range(jump * n_epoch):
        print("Iter {}".format(i))


        # データを分割する
        x = chainer.Variable(np.asarray(
            [d[(jump * j + i) % whole_len] for j in range(batchsize)]))
        t = chainer.Variable(np.asarray(
            [d[(jump * j + i + 1) % whole_len] for j in range(batchsize)]))

        # lossの計算
        loss_i = model(x, t)
        sum_loss += loss_i
        log_loss += loss_i.data

        if (i + 1) % bprop_len == 0:
            print(sum_loss.data / bprop_len)
            model.zerograds()
            sum_loss.backward()
            optimizer.update()

            sum_loss = 0

        if (i + 1) % 10000 == 0:
            print("Training loss: {}".format(log_loss / 10000))
            log_loss = 0

        if (i + 1) % jump == 0:
            epoch += 1
            print("Epoch {}".format(epoch))
            
    # Save File
    net.save(model)
    print("Parameter is saved")