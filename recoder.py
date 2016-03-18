#   encode: utf-8

from datetime import datetime
from pytz import timezone  
import os
import shutil
import subprocess
import time
import wave
import numpy as np
import struct

import chainer
from chainer import optimizers
from chainer import serializers

import encode_net as net

rtmp_url = "rtmpe://fms1.uniqueradio.jp/"
app_url = "?rtmp://fms-base1.mitene.ad.jp/agqr/"
tmp_dir = "tmp"
ffmpeg_path = "ffmpeg"

# ------------------------------
#   Setup
# ------------------------------
model, timestamp = net.load_latest()

optimizer = optimizers.Adam()
optimizer = optimizers.SGD(lr=.1)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

def recode(path):
    print("Recoding to {}".format(path))
    
    result = subprocess.call("rtmpdump --rtmp {} --playpath aandg22 --app {} --timeout 5 --live --flv {} --stop 60".format(
                    rtmp_url, app_url, path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result == 0:
        return True
    else:
        return False

def convert2wav(inpath, outpath):
    print("Converting to wav: {} -> {}".format(inpath, outpath))
    result = subprocess.call("ffmpeg -y -i {} -ac 1 -ar 44100 {}".format(inpath, outpath), 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result == 0:
        return True
    else:
        return False

def convert2float(inpath, outpath):
    print("Converting to float: {} -> {}".format(inpath, outpath))
    #   Convert to float
    wf = wave.open(inpath)

    #   read data
    x = wf.readframes(wf.getnframes())

    #   close file
    wf.close()

    #   convert to [-1, 1]
    x = np.frombuffer(x, dtype='int16') / 32768.0

    #   write
    ff = open(outpath, 'wb')
    ff.write(struct.pack('f' * len(x), *x))
    ff.close()

    return os.path.isfile(outpath)

def convert2power(inpath, outpath):
    print("Converting to power: {} -> {}".format(inpath, outpath))
    
    result = subprocess.call('frame -l 1024 -p 256 < {} | window -l 1024 | fftr -l 1024 -P > {}'.format(inpath, outpath), 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result == 0:
        return True
    else:
        return False

def load(filename):
    fin = open(filename, 'rb')
    buf = fin.read()
    fin.close()

    ffts = []

    for ind in range(0, len(buf), 1024 * 4):
        ffts.append(struct.unpack('f' * 1024, buf[ind:ind+1024*4]))

    return np.asarray(ffts, dtype=np.float32)

def save(model):
    path = os.path.join('encode-model', '{}'.format(int(time.time())))
    serializers.save_hdf5(path, model)

def learn(datapath):
    print("Learning from {}".format(datapath))

    #   Learning
    batchsize = 20
    bprop_len = 30
    n_epoch = 2
    n_refresh = 105

    d = load(datapath)

    whole_len = d.shape[0]
    jump = whole_len // batchsize
    sum_loss = 0
    log_loss = 0

    print("Learning First Layer")
    model.reset_state()
    for i in range(jump):
        x = chainer.Variable(np.asarray(
            [d[(jump * j + i) % whole_len] for j in range(batchsize)]))
        t = chainer.Variable(np.asarray(
            [d[(jump * j + i + 1) % whole_len] for j in range(batchsize)]))

        loss_i = model.layer1(x, t)
        sum_loss += loss_i
        log_loss += loss_i.data

        if (i + 1) % bprop_len == 0:
            model.zerograds()
            sum_loss.backward()
            sum_loss.unchain_backward()
            optimizer.update()

            sum_loss = 0

        if (i + 1) % 10000 == 0:
            print("Training loss: {}".format(log_loss / 10000))
            log_loss = 0

    print("Encoding First Layer")
    model.reset_state()
    enc = np.ndarray(shape=(d.shape[0], 256), dtype=np.float32)
    for i in range(d.shape[0]):
        x = chainer.Variable(np.asarray([d[i]]))
        enc[i] = model.encode1(x).data[0]

    print("Learning Second Layer")
    model.reset_state()
    for i in range(jump):
        x = chainer.Variable(np.asarray(
            [enc[(jump * j + i) % whole_len] for j in range(batchsize)]))
        t = chainer.Variable(np.asarray(
            [enc[(jump * j + i + 1) % whole_len] for j in range(batchsize)]))

        loss_i = model.layer2(x, t)
        sum_loss += loss_i
        log_loss += loss_i.data

        if (i + 1) % bprop_len == 0:
            model.zerograds()
            sum_loss.backward()
            sum_loss.unchain_backward()
            optimizer.update()

            sum_loss = 0

        if (i + 1) % 10000 == 0:
            print("Training loss: {}".format(log_loss / 10000))
            log_loss = 0

    save(model)
    optimizer.lr /= 1.1

def evaluate():
    print("Evaluating")
    evaluator = model.copy()
    evaluator.reset_state()

    total_len = 0

    layer1_loss = 0
    layer2_loss = 0

    for f in os.listdir('evaluation'):
        data = load(os.path.join('evaluation', f))

        total_len += data.shape[0]

        #   Evaluate first layer
        for i in range(data.shape[0]-1):
            x = chainer.Variable(np.asarray([data[i]]), volatile='on')
            t = chainer.Variable(np.asarray([data[i+1]]), volatile='on')

            layer1_loss += evaluator.layer1(x, t).data

        #   Encode to second layer
        evaluator.reset_state()
        enc = np.ndarray(shape=(data.shape[0], 256), dtype=np.float32)
        for i in range(data.shape[0]):
            x = chainer.Variable(np.asarray([data[i]]))
            enc[i] = model.encode1(x).data[0]

        #   Evaluate second layer
        for i in range(enc.shape[0]-1):
            x = chainer.Variable(np.asarray([enc[i]]), volatile='on')
            t = chainer.Variable(np.asarray([enc[i+1]]), volatile='on')

            layer2_loss += evaluator.layer2(x, t).data

    print("Layer 1 evaluation loss {}".format(layer1_loss / total_len))
    print("Layer 2 evaluation loss {}".format(layer2_loss / total_len))

def update_evaluation():
    print("Updating evaluation")
    if np.random.random() < 0.01:
        evaluation_files = os.listdir('evaluation')
        ind = np.random.randint(0, len(evaluation_files))
        filepath = evaluation_files[ind]
        os.remove(os.path.join('evaluation', filepath))
        os.system("mv tmp/*.power evaluation/")

# ------------------------------
#   Loop
# ------------------------------
loop_count = 0
if __name__ == '__main__':
    while True:
        loop_count += 1

        #   Check time
        now = datetime.now(timezone('Japan'))
        #if now.hour == 4 or now.hour == 5:
        #    time.sleep(60 * 60 * 2)

        loopstart = time.time()

        #   Make sure tmp folder exists
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        os.system("rm {}/*".format(tmp_dir))


        basepath = os.path.join(tmp_dir, "{}".format(int(time.time())))

        print("")
        print("---------- Process of {} ----------".format(basepath))
        start = time.time()
        if not recode(basepath + '.flv'):
            continue
        print("Done: {} sec".format(time.time() - start))

        start = time.time()
        if not convert2wav(basepath + '.flv', basepath + '.wav'):
            continue
        print("Done: {} sec".format(time.time() - start))

        start = time.time()
        if not convert2float(basepath + '.wav', basepath + '.float'):
            continue
        print("Done: {} sec".format(time.time() - start))

        start = time.time()
        if not convert2power(basepath + '.float', basepath + '.power'):
            continue
        print("Done: {} sec".format(time.time() - start))

        start = time.time()
        learn(basepath + '.power')
        print("Done: {} sec".format(time.time() - start))

        start = time.time()
        evaluate()
        print("Done: {} sec".format(time.time() - start))

        if loop_count % 10:
            start = time.time()
            update_evaluation()
            print("Done: {} sec".format(time.time() - start))

        print("Time: {} sec".format(time.time() - loopstart))