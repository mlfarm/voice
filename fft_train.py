#   encode: utf8

import time
import math

import numpy as np
import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda

import data
import fft_net as net

#   Warningを表示する
import warnings
#warnings.simplefilter("error")

# ========== モデルのロード ==========
model = net.load_latest()
cuda.get_device(0).use()
model.to_gpu()
xp = cuda.cupy

# ========== 最適化 ==========
optimizer = optimizers.Adam()
optimizer.setup(model)

# ========== 評価ルーチン ==========
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

# ========== 学習 ==========
# 基本設定
batchsize = 20
bprop_len = 30
n_epoch = 10
n_refresh = 13

for refresh in range(n_refresh):
    # ========== データのロード ==========
    print("Loading Data")
    train_data, test_data = np.split(data.load_log_fft(), [8000])
    print("Done")
    
    # 学習データの大きさ
    whole_len = train_data.shape[0]

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
        # データを分割する
        x = chainer.Variable(xp.asarray(
            [train_data[(jump * j + i) % whole_len] for j in range(batchsize)]))
        t = chainer.Variable(xp.asarray(
            [train_data[(jump * j + i + 1) % whole_len] for j in range(batchsize)]))

        # lossの計算
        loss_i = model(x, t)
        sum_loss += loss_i
        log_loss += loss_i.data

        if (i + 1) % bprop_len == 0:
            print(sum_loss.data)
            model.zerograds()
            sum_loss.backward()
            sum_loss = 0
            optimizer.update()

        if (i + 1) % 10000 == 0:
            print("Training loss: {}".format(log_loss / 10000))
            log_loss = 0

        if (i + 1) % jump == 0:
            epoch += 1

            valid_loss = evaluate(test_data)

            print("Evaluation")
            print("epoch {} validation loss: {}".format(epoch, valid_loss))
            
    # Save File
    net.save(model)
    print("Parameter is saved")