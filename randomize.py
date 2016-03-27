import voice
import struct
import numpy as np
import threading
import time

session = voice.Session()
aliases = session.all_alias()

threadCount = 0

def thread(alias, files):
    global threadCount
    threadCount += 1
    print("Alias: {}".format(alias))

    data = []

    for f in files:
        print(" File: {}".format(f + '.power.encode'))
        data.extend(voice.encode(voice.load_power('data/power/' + f + '.power')))

    fout = open('data/speaker/' + alias + '.encode', 'wb')
    x = np.asarray(data, dtype=np.float32).ravel()
    fout.write(struct.pack('f' * len(x), *x))
    threadCount -= 1

for alias in aliases:
    files = session.get_file_by_alias(alias)
    t = threading.Thread(target=thread, args=(alias, files))
    t.start()  

    while threadCount > 10:
        time.sleep(1)