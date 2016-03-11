
import numpy as np
import os
import wave
import struct
import chainer
import fft_net as net
import subprocess

model, timestamp = net.load_latest()

if __name__ == '__main__':
    #   List all files need to process
    files = os.listdir('data/pending')

    #   Iterate all files
    for f in files:
        print("parsing {}".format(f))

        #   Get basename
        basename = '.'.join(f.split('.')[:-1])

        #   Move the pending file to wave file
        print("    Writing wave file")
        os.rename(os.path.join('data/pending', f), os.path.join('data/wave', basename + '.wav'))

        # ------------------------------
        #   convert to float
        # ------------------------------
        print("    Convert to float")
        wf = wave.open(os.path.join('data/wave', basename + '.wav'))
        ff = open(os.path.join('data/float', basename + '.float'), 'wb')

        #   read data
        x = wf.readframes(wf.getnframes())

        #   close file
        wf.close()

        #   convert to [-1, 1]
        x = np.frombuffer(x, dtype='int16') / 32768.0

        #   write
        ff.write(struct.pack('f' * len(x), *x))

        # ------------------------------
        #   power spectrum
        # ------------------------------
        print("    FFT")
        fftProc = subprocess.Popen('frame -l 1024 -p 256 < {} | window -l 1024 | fftr -l 1024 -P > {}'
            .format(os.path.join('data/float', basename + '.float'), os.path.join('data/power', basename + '.power')), stdout=subprocess.PIPE, shell=True)
        fftProc.wait()

        # ------------------------------
        #   encode
        # ------------------------------
        print("    Encode")
        fin = open(os.path.join('data/power', basename + '.power'), 'rb')
        fout = open(os.path.join('data/encode', basename + '.bin'), 'wb')

        buf = fin.read(1024 * 4)
        while len(buf) != 0:
            x = chainer.Variable(np.asarray([struct.unpack('f' * 1024, buf)], dtype=np.float32))
            enc = model.encode(x).data[0]
            fout.write(struct.pack('f' * 64, *enc))
            fout.flush()

            buf = fin.read(1024 * 4)
        fin.close()