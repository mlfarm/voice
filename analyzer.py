import struct
import numpy as np

vecfile = open('speaker/speaker-average/sigma_w_nishi', 'rb')

vec = np.asarray(struct.unpack('f' * 16, vecfile.read()))

datafile = open('speaker/labeled-speaker/sigma_w_34', 'rb')

data = []

buf = datafile.read()

whole_len = int(len(buf) / 16)
norms = np.ndarray(shape=(whole_len, ))
count = 0
for i in range(0, len(buf), 64):
    d = np.asarray(struct.unpack('f' * 16, buf[i:i+64]))

    diff = vec - d
    norm = diff.dot(diff)
    norms[count] = norm
    count +=1

print("Mean: {}".format(np.mean(norms)))
print("STD : {}".format(np.std(norms)))
print("Max : {}".format(np.max(norms)))