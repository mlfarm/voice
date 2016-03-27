import voice
import struct
import os
import threading
import time

db = voice.SpeakerDatabase()

threadCount = 0

def threadRun(alias):
    global db, threadCount

    print("Writing to {}".format('speaker/labeled/' + alias))
    fout = open('speaker/labeled/' + alias, 'wb')

    files = db.listFile(alias)

    for f in files:
        path = os.path.join('speaker/encode', f + '.encode')
        fin = open(path, 'rb')

        #   Copy
        fout.write(fin.read())

        #   Close
        fin.close()

    fout.close()
    threadCount -= 1


alias = db.listAlias()

for a in alias:
    print(a)
    t = threading.Thread(target=threadRun, args=(a, ))
    t.start()
    threadCount += 1

    while threadCount > 20:
        time.sleep(1)