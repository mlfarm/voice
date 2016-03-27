import os
import voice
import subprocess
files = os.listdir('data/copy')

for f in files:
    print(f)
    voice.convert2power('data/copy/' + f, 'data/power/' + f + '.power')
    os.remove('data/copy/' + f)