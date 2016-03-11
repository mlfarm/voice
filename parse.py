
import numpy as np
import os
import wave
import struct
import chainer
import fft_net as net
import subprocess

model, timestamp = net.load_latest()

def raw2float(infile, output):
    """
        convert raw file to float file
    """
    wf = wave.open(infile)

    #   read data
    x = wf.readframes(wf.getnframes())

    #   close file
    wf.close()

    #   convert to [-1, 1]
    x = np.frombuffer(x, dtype='int16') / 32768.0

    #   write
    ff = open(outfile, 'wb')
    ff.write(struct.pack('f' * len(x), *x))
    ff.close()

def float2power(infile, outfile):
    """
        take power spectrum of float file
    """
    fftProc = subprocess.Popen('frame -l 1024 -p 256 < {} | window -l 1024 | fftr -l 1024 -P > {}'
        .format(infile, outfile), stdout=subprocess.PIPE, shell=True)
    fftProc.wait()

def encode(infile, outfile):
    """
        encode power spectrum
            using latest parameter
    """
    model.reset_state()
    fin = open(infile, 'rb')
    fout = open(outfile, 'wb')

    buf = fin.read(1024 * 4)
    while len(buf) != 0:
        x = chainer.Variable(np.asarray([struct.unpack('f' * 1024, buf)], dtype=np.float32), volatile='on')
        enc = model.encode(x).data[0]
        fout.write(struct.pack('f' * 64, *enc))
        fout.flush()

        buf = fin.read(1024 * 4)
    fin.close()
    fout.close()

if __name__ == '__main__':
    #   List all files need to process
    files = os.listdir('data/pending')
    files.sort()

    #   Iterate all files
    count = 0
    for f in files:
        count += 1
        print("{}/{} parsing {}".format(count, len(files), f))

        #   Get basename
        basename = '.'.join(f.split('.')[:-1])

        #   Move the pending file to wave file
        os.rename(os.path.join('data/pending', f), os.path.join('data/wave', basename + '.wav'))

        # ------------------------------
        #   convert to float
        # ------------------------------
        raw2float(os.path.join('data/wave', basename + '.wav'), os.path.join('data/float', basename + '.float'))

        # ------------------------------
        #   power spectrum
        # ------------------------------
        float2power(os.path.join('data/float', basename + '.float'), os.path.join('data/power', basename + '.power'))

        # ------------------------------
        #   encode
        # ------------------------------
        encode(os.path.join('data/power', basename + '.power'), os.path.join('data/encode', "{}_{}.bin".format(timestamp, basename))