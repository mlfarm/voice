import argparse

import voice
import os
import threading
import time

files = os.listdir('speaker/raw')

threadCount = 0
def parse(f):
    global threadCount

    start = time.time()
    voice.convert2wav(os.path.join('speaker/raw', f), os.path.join('speaker/wav', f + '.wav'))
    voice.convert2float(os.path.join('speaker/wav', f + '.wav'), os.path.join('speaker/float', f + '.float'))
    voice.convert2power(os.path.join('speaker/float', f + '.float'), os.path.join('speaker/power', f + '.power'))
    print("Process {}".format(time.time() - start))
    threadCount -= 1

for f in files:
    print("Parsing {}".format(f))

    start = time.time()
    voice.convert2wav(os.path.join('speaker/raw', f), os.path.join('speaker/wav', f + '.wav'))
    voice.convert2float(os.path.join('speaker/wav', f + '.wav'), os.path.join('speaker/float', f + '.float'))
    voice.convert2power(os.path.join('speaker/float', f + '.float'), os.path.join('speaker/power', f + '.power'))
    print("Process {}".format(time.time() - start))