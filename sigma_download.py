#   encode: utf-8
import voice
import time
import os
import subprocess
import codecs

db = voice.SpeakerDatabase()

def load(alias, audioURL):
    #   Get Basename
    basename = '.'.join(os.path.basename(audioURL).split('.')[:-1])

    print("Loading {}".format(basename))
    #   Download
    res = subprocess.call('wget {} -O {} --timeout=60'.format(audioURL, os.path.join('tmp', basename + '.mp3')), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if res != 0:
        return False

    #   Add this file
    db.addFile(alias, os.path.join('tmp', basename + '.mp3'))

    return True

def load_talent(talent, name):
    #   Register this talent
    db.register('sigma_' + talent, name)

    #   フリートーク
    time.sleep(1)
    load('sigma_' + talent, 'http://www.sigma7.co.jp/profile/mp3/{}_f.mp3'.format(talent))

    #   ナレーション
    index = 1
    while True:
        time.sleep(1.0)
        if load('sigma_' + talent, 'http://www.sigma7.co.jp/profile/mp3/{}_{:0>2}.mp3'.format(talent, index)):
            index += 1
        else:
            break

    #   セリフ
    index = 1
    while True:
        time.sleep(1.0)
        if load('sigma_' + talent, 'http://www.sigma7.co.jp/profile/mp3/{}_se{:0>2}.mp3'.format(talent, index)):
            index += 1
        else:
            break

if __name__ == '__main__':
    pf = codecs.open('sigma_profile.txt', 'r', 'utf-8')
    for talent in pf:
        talent = talent.split()

        time.sleep(2.0)

        print("Loading {}:{}".format(talent[0], talent[1]))
        load_talent(talent[0], talent[1])
    pf.close()