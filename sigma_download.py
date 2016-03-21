#   encode: utf-8
import voice

db = voice.SpeakerDatabase()

def load(alias, audioURL):
    #   Get Basename
    basename = '.'.join(os.path.basename(audioURL).split('.')[:-1])

    #   Download
    res = subprocess.call('wget {} -O {}'.format(audioURL, os.path.join('tmp', basename + '.mp3')), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if res != 0:
        return False

    #   Add this file
    db.addFile(alias, os.path.join('tmp', basename + '.mp3'))

    return True

def load_talent(talent):
    #   Register this talent
    db.register('sigma_' + talent)

    #   フリートーク
    time.sleep(1)
    load('sigma_' + talent, 'http://www.sigma7.co.jp/profile/mp3/{}_f.mp3'.format(talent))

    #   ナレーション
    index = 1
    while True:
        time.sleep(1.0)
        if load('http://www.sigma7.co.jp/profile/mp3/{}_{0:0>2}.mp3'.format(talent, index)):
            index += 1
        else:
            break
