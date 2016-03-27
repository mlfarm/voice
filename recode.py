import voice
import os
import time
import datetime
import threading

tmp_dir = 'tmp'
speaker_dir = 'speaker/speaker'
threadCount = 0

def threadRun(basename):
    global threadCount

    start = time.time()

    voice.convert2wav(os.path.join(tmp_dir, basename + '.flv'), os.path.join(tmp_dir, basename + '.wav'))
    voice.convert2float(os.path.join(tmp_dir, basename + '.wav'), os.path.join(tmp_dir, basename + '.float'))
    voice.convert2power(os.path.join(tmp_dir, basename + '.float'), os.path.join(tmp_dir, basename + '.power'))
    voice.encode(os.path.join(tmp_dir, basename + '.power'), os.path.join(tmp_dir, basename + '.encode'))
    voice.speaker_encode(os.path.join(tmp_dir, basename + '.encode'), os.path.join(speaker_dir, basename + '.speaker'))

    #   Removing Template files
    os.remove(os.path.join(tmp_dir, basename + '.flv'))
    os.remove(os.path.join(tmp_dir, basename + '.wav'))
    os.remove(os.path.join(tmp_dir, basename + '.float'))
    os.remove(os.path.join(tmp_dir, basename + '.power'))
    os.remove(os.path.join(tmp_dir, basename + '.encode'))

    threadCount -= 0

    print("Process Ended: {}".format(time.time() - start))


while True:
    start = time.time()

    #   Get current time
    now = datetime.datetime.now()

    #   Get Basename
    basename = 'agqr_{:4}_{:0>2}_{:0>2}_{:0>2}_{:0>2}_{:0>2}'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)

    #   Start recoding
    voice.recodeAGQR(30, os.path.join(tmp_dir, basename + '.flv'))

    #   Process recoding in different thread
    t = threading.Thread(target=threadRun, args=(basename, ))
    t.start();