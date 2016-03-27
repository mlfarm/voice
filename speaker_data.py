import struct
import os
import numpy as np

label_files = os.listdir('speaker/labeled')

labels = []
label_size = []

for label in label_files:
    labels.append(open(os.path.join('speaker/labeled', label), 'rb'))
    label_size.append(int(os.path.getsize(os.path.join('speaker/labeled', label)) / 256))

def getFrame(fp, id):
    fp.seek(id * 64 * 4)
    return struct.unpack('f' * 64, fp.read(64 * 4))

def load(n):
    difference = []
    x0 = []
    x1 = []
    for i in range(n):
        ind0 = np.random.randint(0, len(labels))

        fp0 = labels[ind0]
        size0 = label_size[ind0]

        frame0 = getFrame(fp0, np.random.randint(0, size0))

        if i & 2 == 0:
            frame1 = getFrame(fp0, np.random.randint(0, size0))
            d = 1
        else:
            d = 0
            x = list(range(len(labels)))
            x.remove(ind0)
            k = np.random.choice(x)
            frame1 = getFrame(labels[k], np.random.randint(0, label_size[k]))

        difference.append(d)
        x0.append(frame0)
        x1.append(frame1)

    return difference, x0, x1

def close():
    for label in labels:
        label.close()