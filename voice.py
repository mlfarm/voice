import struct
import numpy as np
import os
import shutil
import subprocess

import chainer

import wave

import tmp

import encode_net

#   Database files
speaker_index = 'data/default.speaker'
file_index = 'data/default.file'

#   Encode model
encode_model, stamp = encode_net.load_latest(False)

def recodeAGQR(length, outpath):
    print("Recording to {}".format(outpath))

    result = subprocess.call(
        "rtmpdump --rtmp rtmpe://fms1.uniqueradio.jp/ --playpath aandg22 --app ?rtmp://fms-base1.mitene.ad.jp/agqr/ --timeout 5 --live --flv {} --stop {}".format(
        outpath, length
    ), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    return True if result == 0 else False

def convert2power(audiofile, outputfile):
    """
        Load any type of audio file and convert to power spectrum file
    """
    #   tmp basename
    basename = os.path.basename(audiofile)

    #   tmp Session
    session = tmp.session()

    #   FFMPEG
    print("Converting to Wav")
    tmp_wav = session.create(basename + '.wav')
    subprocess.call('ffmpeg -y -i {} -ac 1 -ar 44100 {}'.format(audiofile, tmp_wav), shell=True)

    #   Convert to float
    print("Converting to float")
    wf = wave.open(tmp_wav)
    x = wf.readframes(wf.getnframes())
    wf.close()

    x = np.frombuffer(x, dtype=np.int16)
    x = x.astype(dtype=np.float32)

    tmp_float = session.create(basename + '.float')
    fout = open(tmp_float, 'wb')
    fout.write(struct.pack('f' * len(x), *x))
    fout.close()

    #   Take Power
    print("Converting to")
    subprocess.call('frame -l 1024 -p 256 < {} | window -l 1024 | fftr -l 1024 -P > {}'.format(tmp_float, outputfile),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    session.reset()

def load_power(filepath):
    fin = open(filepath, 'rb')
    buf = fin.read()
    fin.close()

    x = np.frombuffer(buf, dtype=np.float32)

    return np.log(x.reshape((len(x) // 1024, 1024)) + 1)

#   Encoding
def encode_latest_model(train=True):
    return encode_net.load_latest(train)

def encode(power_data):
    global encode_model

    encode_model.reset_state()

    #   Placeholder
    enc = np.ndarray(shape=(power_data.shape[0], 64), dtype=np.float32)

    #   Encode
    for i in range(power_data.shape[0]):
        x = chainer.Variable(np.asarray(power_data[i:i+1]), volatile='on')

        enc[i] = encode_model.encode(x).data

    return enc

def encode_evaluate(power_data):
    global encode_model

    encode_model.reset_state()
    
    sum_loss = 0

    #   Encode
    for i in range(power_data.shape[0] - 1):
        x = chainer.Variable(np.asarray(power_data[i:i+1]), volatile='on')
        t = chainer.Variable(np.asarray(power_data[i+1:i+2]), volatile='on')
        sum_loss += encode_model(x, t)

    return sum_loss.data / power_data.shape[0]

#   Session class
class Session(object):
    def __init__(self, speaker_index='data/default.speaker', file_index='data/default.file'):
        self.speaker = open(speaker_index, 'r+')
        self.file = open(file_index, 'r+')

    def close(self):
        """
            Close the session
        """
        self.speaker.close()
        self.file.close()

    def find_by_alias(self, alias):
        #   Move to the begging of file
        self.speaker.seek(0, 0)

        id = 0

        for line in self.speaker:
            speaker = line.split()
            if speaker[0] == alias:
                return id, speaker[1]
            id += 1

    def get_file_by_alias(self, alias):
        self.file.seek(0, 0)

        files = []
        for line in self.file:
            f = line.split()
            if f[0] == alias:
                files.append(f[1])

        return files

    def all_alias(self):
        self.speaker.seek(0, 0)

        alias = []

        for line in self.speaker:
            speaker = line.split()
            alias.append(speaker[0])

        return alias

    def add_alias(self, alias, name):
        """
            Add alias
        """
        self.speaker.seek(2, 0)
        self.speaker.write("{} {}\n".format(alias, name))

    def add_file(self, alias, filepath):
        """
            Add file
        """
        basename = os.path.basename(filepath)

        #   Copy file
        shutil.copy(filepath, os.path.join('data/raw', basename))

        #   Convert 2 power
        subprocess.call('python convert2power.py {} {}'.format(os.path.join('data/raw', basename), os.path.join('data/power', basename + '.power')),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        
        #   Write to index
        self.file.seek(0, 0)
        self.file.write("{} {}\n".format(alias, basename))

def copy(origin, to):
    shutil.copy(origin + '.speaker', to + '.speaker')
    shutil.copy(origin + '.file', to + '.file')