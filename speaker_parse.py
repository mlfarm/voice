import voice
import os
import subprocess
import threading
import time

def threadRun(input, output):
    global threadCount
    subprocess.call("python encode.py {} {}".format(input, output), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    threadCount -= 1

files = os.listdir('speaker/power')

threadCount = 0
p = []
index = 0

for f in files:
    print("Encoding: {}".format(f))
    #subprocess.call("python encode.py {} {}".format(os.path.join('speaker/power', f), os.path.join('speaker/encode', '.'.join(f.split('.')[:-1]) + ".encode")))
    t = threading.Thread(target=threadRun, args=(os.path.join('speaker/power', f), os.path.join('speaker/encode', '.'.join(f.split('.')[:-1]) + ".encode")))
    t.start()
    threadCount += 1

    while threadCount > 20:
        time.sleep(1)
