import os

#   Location of tmp file
tmp_dir = 'tmp'

#   Session class
class Session(object):
    def __init__(self):
        self.created = []

    def create(self, filename):
        """Create tmp file
        """
        self.created.append(filename)
        return os.path.join(tmp_dir, filename)

    def reset(self):
        for f in self.created:
            os.remove(os.path.join(tmp_dir, f))

def session():
    return Session()

def listall():
    return os.listdir(tmp_dir)

def eraseall():
    files = os.listdir(tmp_dir)

    for f in files:
        os.remove(os.path.join(tmp_dir, f))
