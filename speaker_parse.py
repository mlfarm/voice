import voice
import os

files = os.listdir('speaker/raw')

for f in files:
    basename = '.'.join(os.path.basename(f).split('.')[:-1])

    voice.convert2wav(os.path.join('speaker/raw', f), os.path.join('speaker/wav', basename + '.wav'))

for f in files:
    basename = '.'.join(os.path.basename(f).split('.')[:-1])

    voice.convert2float(os.path.join('speaker/wav', basename + '.wav'), os.path.join('speaker/float', basename + '.float'))

for f in files:
    basename = '.'.join(os.path.basename(f).split('.')[:-1])

    voice.convert2power(os.path.join('speaker/float', basename + '.float'), os.path.join('speaker/power', basename + '.power'))

for f in files:
    basename = '.'.join(os.path.basename(f).split('.')[:-1])
    
    voice.encode(os.path.join('speaker/power', basename + '.power'), os.path.join('speaker/encode', basename))