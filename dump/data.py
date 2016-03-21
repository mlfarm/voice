# encode: utf-8

import numpy as np 
import os
import struct

# Directories
d_preparedFFT = 'fft-prep'
d_preparedLogFFT = 'fft-log-prep'

# Size of One Entity
N = 10000

# indexes
fft_index = 0
log_fft_index = 0

def prepare_fft():
    """
        Prepare FFT Spectrum for AE
    """
    ffts = np.ndarray((N, 1024), dtype=np.float32)

    index = 0

    fcount = 0

    count = 0

    file_list = os.listdir('fft')
    perm = np.random.permutation(len(file_list))

    for i in perm:
        count += 1
        f = file_list[i]

        print(count, f)

        fin = open(os.path.join('fft', f), 'rb')

        buf = fin.read()

        fin.close()

        for ind in range(0, len(buf), 1024 * 4):
            ffts[index] = np.asarray(struct.unpack('f' * 1024, buf[ind:ind+1024*4])).astype(np.float32)

            index += 1

            if index % N == 0:
                print("========== SAVING ==========")
                np.save('fft-prep/{}'.format(fcount), ffts)
                fcount += 1
                index = 0

def prepare_log_fft():
    ffts = np.ndarray((N, 1024), dtype=np.float32)

    index = 0

    fcount = 0

    count = 0

    for i in perm:
        count += 1
        f = file_list[i]

        print(count, f)

        fin = open(os.path.join('../fft', f), 'rb')

        buf = fin.read()

        fin.close()

        for ind in range(0, len(buf), 1024 * 4):
            ffts[index] = np.log(1+np.asarray(struct.unpack('f' * 1024, buf[ind:ind+1024*4]))).astype(np.float32)

            index += 1

            if index % N == 0:
                print("========== SAVING ==========")
                np.save('../fft-log-prep/{}'.format(fcount), ffts)
                fcount += 1
                index = 0

def load_fft(index=None):
    """
        Load Prepared FFT Spectrum
    """
    global fft_index
    if index is None:
        f = d_preparedFFT + '/{}.npy'.format(fft_index)
        fft_index += 1
    else:
        f = d_preparedFFT + '/{}.npy'.format(index)

    return np.load(f)

def load_log_fft(index=None):
    """
        Load Prepared Log FFT Spectrum
    """
    global log_fft_index
    if index is None:
        f = d_preparedLogFFT + '/{}.npy'.format(log_fft_index)
        log_fft_index += 1
    else:
        f = d_preparedLogFFT + '/{}.npy'.format(index)

    return np.load(f)

