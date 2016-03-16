#   encode: utf-8
import os
import subprocess
import time
import wave
import numpy as np
import struct

import chainer
from chainer import optimizers
from chainer import serializers

import encode_net as net

# ------------------------------
#   Configuration
# ------------------------------
#   Recording
#rtmp_path = "C:\\Users\\Jin\\rtmpdump\\rtmpdump.exe"
rtmp_path = "rtmpdump"
rtmp_url = "rtmpe://fms1.uniqueradio.jp/"
app_url = "?rtmp://fms-base1.mitene.ad.jp/agqr/"
#tmp_dir = "C:\\Users\\Jin\\dev\\mlfarm\\agqr\\tmp"
tmp_dir = "tmp"
#ffmpeg_path = "C:\\Users\\Jin\\ffmpeg\\bin\\ffmpeg.exe"
ffmpeg_path = "ffmpeg"

#   Learning
batchsize = 20
bprop_len = 30
n_epoch = 2
n_refresh = 105

# ------------------------------
#   Functions
# ------------------------------
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

# ------------------------------
#   Setup
# ------------------------------
model, timestamp = net.load_latest()

optimizer = optimizers.Adam()
optimizer.setup(model)

while True:
    stamp = int(time.time())

    #   Recode
    subprocess.call("{} --rtmp {} --playpath aandg22 --app {} --timeout 5 --live --flv {}/{}.flv --stop 60".format(
        rtmp_path, rtmp_url, app_url, tmp_dir, stamp), stdout=subprocess.PIPE, shell=True)

    #   Convert to wav
    subprocess.call("{} -y -i {}/{}.flv -ac 1 -ar 44100 {}/{}.wav".format(ffmpeg_path, tmp_dir, stamp, tmp_dir, stamp), stdout=subprocess.PIPE, shell=True)

    #   Convert to float
    wf = wave.open("{}/{}.wav".format(tmp_dir, stamp))

    #   read data
    x = wf.readframes(wf.getnframes())

    #   close file
    wf.close()

    #   convert to [-1, 1]
    x = np.frombuffer(x, dtype='int16') / 32768.0

    #   write
    ff = open("{}/{}.float".format(tmp_dir, stamp), 'wb')
    ff.write(struct.pack('f' * len(x), *x))
    ff.close()

    #   Convert to fft
    subprocess.call('frame -l 1024 -p 256 < {}/{}.float | window -l 1024 | fftr -l 1024 -P > {}/{}.power'.format(tmp_dir, stamp, tmp_dir, stamp), stdout=subprocess.PIPE, shell=True)

    #   Learn
    d = load("{}/{}.power".format(tmp_dir, stamp))

    whole_len = d.shape[0]
    jump = whole_len // batchsize
    sum_loss = 0
    log_loss = 0

    for i in range(jump):
        x = chainer.Variable(np.asarray(
            [d[(jump * j + i) % whole_len] for j in range(batchsize)]))
        t = chainer.Variable(np.asarray(
            [d[(jump * j + i + 1) % whole_len] for j in range(batchsize)]))

        loss_i = model(x, t)
        sum_loss += loss_i
        log_loss += loss_i.data

        if (i + 1) % bprop_len == 0:
            print(sum_loss.data / bprop_len)
            model.zerograds()
            sum_loss.backward()
            accum_loss.unchain_backward()
            optimizer.update()

            sum_loss = 0

        if (i + 1) % 10000 == 0:
            print("Training loss: {}".format(log_loss / 10000))
            log_loss = 0

    save(model)

    #   Erase
    os.remove("{}/{}.flv".format(tmp_dir, stamp))
    os.remove("{}/{}.wav".format(tmp_dir, stamp))
    os.remove("{}/{}.float".format(tmp_dir, stamp))
    os.remove("{}/{}.power".format(tmp_dir, stamp))
