#   encode: utf-8

import numpy as np
import chainer
from chainer import optimizers
from chainer import serializers

import encode_data as data
import encode_net as net

print("start")
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
        x = chainer.Variable(np.asarray(dataset[i:i+1]), volatile='on')
        t = chainer.Variable(np.asarray(dataset[i+1:i+2]), volatile='on')

        loss = evaluator(x, t)

        sum_loss += loss.data

    return sum_loss / dataset.shape[0]

#   Train configuration
batchsize = 20
bprop_len = 30
n_epoch = 10
n_refresh = 105

#   Train
for refresh in range(n_refresh):
    print("Loading Data")
    d = data.load(20)
    print("Done {}".format(d.shape[0]))

    d_train, d_test = np.split(d, [70000])


    # 学習データの大きさ
    whole_len = d_train.shape[0]

    # 並列学習の切れ目
    jump = whole_len // batchsize

    # lossの合計
    sum_loss = 0

    # ログ用のloss
    log_loss = 0

    # epoch
    epoch = 0

    count = 0
    # 開始
    print("Iterate: {}".format(jump * n_epoch))
    for i in range(jump * n_epoch):
        count += 1


        # データを分割する
        x = chainer.Variable(np.asarray(
            [d_train[(jump * j + i) % whole_len] for j in range(batchsize)]))
        t = chainer.Variable(np.asarray(
            [d_train[(jump * j + i + 1) % whole_len] for j in range(batchsize)]))

        # lossの計算
        loss_i = model(x, t)
        sum_loss += loss_i
        log_loss += loss_i.data
        #print(loss_i.data)

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            model.zerograds()
            sum_loss.backward()
            sum_loss.unchain_backward()  # truncate
            sum_loss = 0
            optimizer.update()

        if (i + 1) % jump == 0:
            print("Training loss: {}".format(log_loss / count))
            log_loss = 0
            count = 0
            epoch += 1
            print("Evaluate")
            print("Evaluation loss: {}".format(evaluate(d_test)))
            print("Epoch {}".format(epoch))
            
    # Save File
    net.save(model)
    print("Parameter is saved")